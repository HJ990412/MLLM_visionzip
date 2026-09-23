"""Pure correctness fixtures for ReKV-Chunk25's internal retrieval adapter.

Reference: Becomebright/ReKV commit
1fd9a3dbf5dbff7f27069ae2f4463674c495e830, especially
``kv_cache_manager.py`` L172-180, 422-468, 600-608;
``rekv_attention.py`` L64-129; and ``rope.py`` L107-112. The analytical
RoPE/attention oracle below is independent of mmimpress.rekv helpers and
follows the pinned source's compact positions and init/local masks.
"""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

from mmimpress.rekv import (
    RequestState,
    _ReKVAttentionScope,
    compact_visual_rows,
    mean_head_vector,
    official_dot_scores,
    rekv_attention_output,
    select_chunks,
)


def _rotary(head_dim: int = 4):
    config = LlamaConfig(
        hidden_size=head_dim, num_attention_heads=1,
        num_key_value_heads=1, num_hidden_layers=1,
        rope_theta=10000.0, max_position_embeddings=256,
    )
    return LlamaRotaryEmbedding(config=config)


def _manual_rotate(x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """RoPE equation with the LLaMA half-rotation convention, not HF helper."""
    d = x.shape[-1]
    assert d % 2 == 0
    inv = 10000.0 ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    angle = position.to(torch.float64)[:, None] * inv[None, :]
    angle = torch.cat((angle, angle), dim=-1).to(x.dtype)
    c, s = angle.cos()[None, None], angle.sin()[None, None]
    half = d // 2
    rotated_half = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * c + rotated_half * s


def _attention_oracle(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      n_local: int, n_init: int) -> torch.Tensor:
    """Independent two-branch, shared-softmax reference from pinned source.

    Official ``rekv_attention_forward`` feeds a right-aligned compact local
    sequence into RoPE, and, only beyond the window, an unrotated initial-K
    branch with Q rotated to the ``n_local-1`` distance ceiling. Its Torch
    multi-stage attention joins both branches before softmax.
    """
    nq, nk = q.shape[-2], k.shape[-2]
    st = max(0, nk - nq - n_local)
    local_k, local_v = k[..., st:, :], v[..., st:, :]
    local_len = local_k.shape[-2]
    q_pos = torch.arange(local_len - nq, local_len)
    k_pos = torch.arange(local_len)
    q_rot = _manual_rotate(q, q_pos)
    k_rot = _manual_rotate(local_k, k_pos)
    dist = q_pos[:, None] - k_pos[None, :]
    allowed = (dist >= 0) & (dist < n_local)
    logits = (q_rot @ k_rot.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    logits = logits.masked_fill(~allowed[None, None], float('-inf'))
    pieces = [logits]
    vals = [local_v]
    if nk > n_local and n_init:
        init_q = _manual_rotate(q, torch.full((nq,), n_local - 1))
        init_k, init_v = k[..., :n_init, :], v[..., :n_init, :]
        init_dist = (nk - nq + torch.arange(nq)[:, None]
                     - torch.arange(n_init)[None, :])
        init_allowed = init_dist >= n_local
        init_logits = (init_q @ init_k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        init_logits = init_logits.masked_fill(
            ~init_allowed[None, None], float('-inf'))
        pieces.append(init_logits)
        vals.append(init_v)
    weight = torch.cat(pieces, dim=-1).softmax(dim=-1)
    out = weight @ torch.cat(vals, dim=-2)
    return out.transpose(1, 2).contiguous()


class RepresentativeAndSelectionTests(unittest.TestCase):
    def test_k_and_q_pre_rope_mean_then_all_head_flatten(self):
        # Heads carry disjoint values, so transposing/averaging heads or taking
        # just the three QA probe heads cannot pass this fixture.
        raw = torch.tensor([[[[1., 2.], [3., 4.], [5., 6.]],
                             [[10., 20.], [30., 40.], [50., 60.]]]])
        expected = torch.tensor([[3., 4., 30., 40.]])
        self.assertTrue(torch.equal(mean_head_vector(raw), expected))
        self.assertTrue(torch.equal(mean_head_vector(raw.to(torch.bfloat16)),
                                    expected.to(torch.bfloat16)))
        # Original representative is formed before rotation. Rotated-then-mean
        # is different even for the same values and contiguous positions.
        rotated = _manual_rotate(raw, torch.arange(3))
        self.assertFalse(torch.allclose(mean_head_vector(rotated), expected,
                                        rtol=1e-5, atol=1e-6))
        question = torch.tensor([[[[1., 2.], [5., 6.]],
                                  [[3., 4.], [7., 8.]]]])
        self.assertTrue(torch.equal(mean_head_vector(question),
                                    torch.tensor([[3., 4., 5., 6.]])))

    def test_official_vector_cache_uses_fp32_dot_without_normalization(self):
        q = torch.tensor([[2., 1.]], dtype=torch.bfloat16)
        keys = torch.tensor([[3., 0.], [0., 5.], [4., 4.]],
                            dtype=torch.bfloat16)
        scores = official_dot_scores(q, keys)
        self.assertEqual(scores.dtype, torch.float32)
        self.assertTrue(torch.equal(scores, torch.tensor([6., 5., 12.])))
        cosine = torch.nn.functional.cosine_similarity(
            keys.float(), q.float().expand_as(keys), dim=-1)
        self.assertFalse(torch.allclose(scores, cosine))
        # A large BF16 dot product must be computed after the FP32 cast to
        # avoid FP16/BF16 dot overflow in the official vector-cache path.
        large = official_dot_scores(torch.tensor([400., 300.], dtype=torch.float16),
                                    torch.tensor([[500., 600.]], dtype=torch.float16))
        self.assertTrue(torch.isfinite(large).all())
        self.assertEqual(float(large[0]), 380000.)

    def test_nontie_topk_matches_official_and_assembly_is_source_order(self):
        scores = torch.tensor([0.1, 0.9, -0.3, 0.6, 0.8, 0.2, 0.7, 0.4])
        counts = torch.ones(8, dtype=torch.long)
        ranked, ordered = select_chunks(scores, counts, ratio=0.375)
        self.assertEqual(ranked, torch.topk(scores, 3).indices.tolist())
        self.assertEqual(ranked, [1, 4, 6])
        self.assertEqual(ordered, [1, 4, 6])
        # Scores in a non-source ranking order expose the sort-for-I/O step.
        s2 = torch.tensor([0., 2., 0., 3., 0., 1., 0., 0.])
        ranked2, ordered2 = select_chunks(s2, counts, ratio=0.375)
        self.assertEqual(ranked2, [3, 1, 5])
        self.assertEqual(ordered2, [1, 3, 5])

    def test_tie_break_lower_physical_id_and_budget_rounding(self):
        scores = torch.ones(8)
        counts = torch.ones(8, dtype=torch.long)
        self.assertEqual(select_chunks(scores, counts)[0], [0, 1])
        self.assertEqual(select_chunks(torch.arange(34.).float(),
                                       torch.ones(34, dtype=torch.long))[0],
                         list(range(33, 25, -1)))  # round(8.5) = 8

    def test_zero_valid_chunks_are_not_ranked_but_budget_is_physical(self):
        # QA/Ours count all eight normal physical chunks for the shared 25%
        # budget (two), then exclude structural-only chunks from ranking.
        scores = torch.tensor([100., 3., 99., 2., 98., 1., 97., 0.])
        counts = torch.tensor([0, 2, 0, 2, 0, 2, 0, 2])
        ranked, ordered = select_chunks(scores, counts)
        self.assertEqual(ranked, [1, 3])
        self.assertEqual(ordered, [1, 3])

    def test_insufficient_valid_candidates_fails_closed(self):
        # The shared nominal budget is two; if only one block has spatial
        # tokens, silently selecting one would break the fixed-budget match.
        scores = torch.tensor([100., 3., 99., 2., 98., 1., 97., 0.])
        counts = torch.tensor([0, 2, 0, 0, 0, 0, 0, 0])
        with self.assertRaises((ValueError, AssertionError)):
            select_chunks(scores, counts)

    def test_compaction_keeps_selected_rows_and_separators_once(self):
        rows = compact_visual_rows([3, 0], separators=[1, 5, 10],
                                   v_num=11, chunk_size=3)
        # Chunk 0 -> 0,1,2; chunk 3 -> 9,10; separator 5 stays; no holes.
        self.assertEqual(rows, [0, 1, 2, 5, 9, 10])
        self.assertEqual(len(rows), len(set(rows)))
        self.assertNotIn(3, rows)
        self.assertNotIn(8, rows)
        with self.assertRaises(ValueError):
            compact_visual_rows([4], [], v_num=11, chunk_size=3)


class CompactAttentionTests(unittest.TestCase):
    def setUp(self):
        self.rotary = _rotary()
        self.attn = SimpleNamespace(scaling=1 / math.sqrt(4))
        gen = torch.Generator().manual_seed(11)
        self.k = torch.randn(1, 1, 7, 4, generator=gen)
        self.v = torch.randn(1, 1, 7, 4, generator=gen)
        self.q = torch.randn(1, 1, 2, 4, generator=gen)

    def test_consecutive_compact_rope_and_causal_mask(self):
        k, v = self.k[..., :5, :], self.v[..., :5, :]
        actual, rotated, branch = rekv_attention_output(
            self.attn, self.rotary, self.q, k, v,
            n_local=8, n_init=1, return_rotated_k=True)
        expected = _attention_oracle(self.q, k, v, n_local=8, n_init=1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(branch, 'local_only')
        torch.testing.assert_close(rotated,
                                   _manual_rotate(k, torch.arange(5)),
                                   rtol=1e-5, atol=1e-6)
        # First query must not see second query's K/V, despite both residing
        # in the same compact tensor.
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[..., -1, :] += 1000
        changed_v[..., -1, :] -= 1000
        changed = rekv_attention_output(
            self.attn, self.rotary, self.q, changed_k, changed_v,
            n_local=8, n_init=1)
        torch.testing.assert_close(actual[:, :1], changed[:, :1],
                                   rtol=1e-5, atol=1e-6)

    def test_initial_distance_ceiling_and_local_window_share_softmax(self):
        actual, rotated, branch = rekv_attention_output(
            self.attn, self.rotary, self.q, self.k, self.v,
            n_local=3, n_init=1, return_rotated_k=True)
        expected = _attention_oracle(self.q, self.k, self.v,
                                     n_local=3, n_init=1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(branch, 'local_plus_init')
        self.assertIsNone(rotated)  # full prefix rotation not reusable here
        # Initial V is truly attended: altering it must change output.
        changed_v = self.v.clone()
        changed_v[..., 0, :] += 100
        changed = rekv_attention_output(
            self.attn, self.rotary, self.q, self.k, changed_v,
            n_local=3, n_init=1)
        self.assertGreater((actual - changed).abs().max().item(), 1e-3)


class _FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 1
        self.num_key_value_heads = 1
        self.head_dim = 4
        self.scaling = 0.5
        self.q_proj = nn.Identity()
        self.k_proj = nn.Identity()
        self.v_proj = nn.Identity()
        self.o_proj = nn.Identity()

    def forward(self, hidden_states, **_):
        return hidden_states, None


class _FakeReader:
    def __init__(self, first_layer_value: float):
        self.first_layer_value = first_layer_value
        self.reads = []

    def read_chunks(self, layer, kind, chunks, counter):
        self.reads.append((layer, kind, tuple(chunks)))
        rows = [2 * chunks[0], 2 * chunks[0] + 1]
        value = torch.zeros(2, 1, 4)
        if layer == 0 and kind == 'k':
            value[:, 0, 0] = 1
        if layer == 0 and kind == 'v':
            value[:, 0, 1] = self.first_layer_value
        return rows, value


class MultiLayerDependencyTests(unittest.TestCase):
    def _run_two_layers(self, first_layer_value: float):
        layers = [SimpleNamespace(self_attn=_FakeAttention()) for _ in range(2)]
        rotary = _rotary()
        model = SimpleNamespace(model=SimpleNamespace(
            language_model=SimpleNamespace(rotary_emb=rotary)))
        runner = SimpleNamespace(layers=layers, model=model)
        # Layer 0 selects chunk 0 from dim 0. Layer 1 selects chunk 0 if
        # the first-layer attention output is zero, chunk 1 if that selected
        # payload injects enough dim-1 signal into the residual stream.
        rep = torch.tensor([[[10., 0., 0., 0.], [-10., 0., 0., 0.]],
                            [[1., 0., 0., 0.], [0., 1., 0., 0.]]])
        ctx = SimpleNamespace(
            meta={'valid_spatial_counts': [2, 2], 'newline_idx': [],
                  'v_token_num': 4, 'chunk_size': 2, 'v_token_start': 0},
            k_rep=rep,
            reader=_FakeReader(first_layer_value),
            sys_kv={'k': [torch.empty(1, 0, 4) for _ in range(2)],
                    'v': [torch.empty(1, 0, 4) for _ in range(2)]},
        )
        state = RequestState()
        empty_sep = torch.empty(2, 0, 1, 4)
        state.separator_kv = (empty_sep, empty_sep)
        initial = torch.tensor([[[1., 0., 0., 0.],
                                 [1., 0., 0., 0.]]])
        original_forwards = [layer.self_attn.forward for layer in layers]
        with _ReKVAttentionScope(runner, ctx, state, n_local=8):
            first, _ = layers[0].self_attn(hidden_states=initial)
            next_hidden = initial + first  # decoder residual
            layers[1].self_attn(hidden_states=next_hidden)
        for layer, original in zip(layers, original_forwards):
            self.assertEqual(layer.self_attn.forward, original)
        self.assertEqual(state.retrieval_calls, 2)
        self.assertEqual(len(state.layers), 2)
        self.assertEqual(ctx.reader.reads,
                         [(0, 'k', (0,)), (0, 'v', (0,)),
                          (1, 'k', tuple(state.log[1]['selected_chunk_ids'])),
                          (1, 'v', tuple(state.log[1]['selected_chunk_ids']))])
        self.assertEqual(state.layers[0].prefix_len, 2)
        self.assertEqual(state.layers[1].prefix_len, 2)
        self.assertTrue(all(h.text_raw_k is None for h in state.layers.values()))
        return state

    def test_later_selection_changes_when_previous_selected_v_changes(self):
        without = self._run_two_layers(0.)
        with_visual = self._run_two_layers(10.)
        self.assertEqual(without.log[0]['selected_chunk_ids'], [0])
        self.assertEqual(with_visual.log[0]['selected_chunk_ids'], [0])
        self.assertEqual(without.log[1]['selected_chunk_ids'], [0])
        self.assertEqual(with_visual.log[1]['selected_chunk_ids'], [1])
        self.assertEqual(with_visual.log[0]['compact_positions'], [0, 1])
        self.assertEqual(with_visual.log[0]['actual_attention_key_len'], 4)


class HandoffAndIsolationTests(unittest.TestCase):
    def _setup(self):
        layer = SimpleNamespace(self_attn=_FakeAttention())
        runner = SimpleNamespace(
            layers=[layer],
            model=SimpleNamespace(model=SimpleNamespace(
                language_model=SimpleNamespace(rotary_emb=_rotary()))))
        ctx = SimpleNamespace(
            meta={'valid_spatial_counts': [2, 2], 'newline_idx': [],
                  'v_token_num': 4, 'chunk_size': 2, 'v_token_start': 0},
            k_rep=torch.tensor([[[10., 0., 0., 0.],
                                 [-10., 0., 0., 0.]]]),
            reader=_FakeReader(4.),
            sys_kv={'k': [torch.empty(1, 0, 4)],
                    'v': [torch.empty(1, 0, 4)]},
        )
        state = RequestState()
        empty_sep = torch.empty(1, 0, 1, 4)
        state.separator_kv = (empty_sep, empty_sep)
        return runner, ctx, state

    def test_stage_a_handoff_excludes_question_and_stage_b_reuses_payload(self):
        runner, ctx, state = self._setup()
        attn = runner.layers[0].self_attn
        with _ReKVAttentionScope(runner, ctx, state, n_local=8):
            # Q-only retrieval sees the two SSD rows; its own K/V must not be
            # returned in the raw visual cache.
            attn(hidden_states=torch.tensor([[[5., 0., 0., 0.]]]))
            handoff = state.layers[0]
            self.assertEqual(handoff.prefix_len, 2)
            self.assertTrue(torch.equal(handoff.raw_k[0, 0, :, 0],
                                        torch.tensor([1., 1.])))
            self.assertIsNone(handoff.text_raw_k)
            self.assertEqual(len(ctx.reader.reads), 2)
            source_k = handoff.raw_k.clone()
            source_v = handoff.value.clone()

            state.mode = 'answer'
            suffix = torch.tensor([[[0., 1., 0., 0.],
                                    [0., 0., 1., 0.]]])
            actual, _ = attn(hidden_states=suffix)
            q = suffix[:, None]
            full_k = torch.cat((source_k, q), dim=-2)
            full_v = torch.cat((source_v, q), dim=-2)
            expected = _attention_oracle(q, full_k, full_v,
                                         n_local=8, n_init=0)
            torch.testing.assert_close(actual, expected.reshape(1, 2, 4),
                                       rtol=1e-5, atol=1e-6)
            self.assertEqual(len(ctx.reader.reads), 2)
            self.assertEqual(handoff.text_raw_k.shape[-2], 2)

            # Decode appends only the new token. It does not reread selected
            # visual KV or mutate persistent/retrieved source tensors.
            attn(hidden_states=torch.tensor([[[0., 0., 0., 1.]]]))
            self.assertEqual(handoff.text_raw_k.shape[-2], 3)
            self.assertEqual(len(ctx.reader.reads), 2)
            self.assertTrue(torch.equal(handoff.raw_k, source_k))
            self.assertTrue(torch.equal(handoff.value, source_v))

    def test_installed_llama_attention_adapter_interface(self):
        # A hand-written fake can accidentally provide attributes that the
        # installed Transformers attention module does not have. Exercise the
        # exact 4.57.x LlamaAttention object used by LLaVA-NeXT instead.
        runner, ctx, state = self._setup()
        config = LlamaConfig(
            hidden_size=4, num_attention_heads=1,
            num_key_value_heads=1, num_hidden_layers=1,
            max_position_embeddings=256,
        )
        attn = LlamaAttention(config=config, layer_idx=0)
        runner.layers[0].self_attn = attn
        with torch.no_grad(), _ReKVAttentionScope(runner, ctx, state):
            output, _ = attn(hidden_states=torch.tensor([[[5., 0., 0., 0.]]]))
        self.assertEqual(tuple(output.shape), (1, 1, 4))
        self.assertEqual(state.retrieval_calls, 1)

    def test_scope_restores_attention_after_exception(self):
        runner, ctx, state = self._setup()
        attn = runner.layers[0].self_attn
        original = attn.forward
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with _ReKVAttentionScope(runner, ctx, state):
                raise RuntimeError('fixture')
        self.assertEqual(attn.forward, original)


if __name__ == '__main__':
    unittest.main()
