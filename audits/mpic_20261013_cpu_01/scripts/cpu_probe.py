"""Fresh, bounded CPU audit of the unmodified MPIC implementation.

Random 3-layer Llama, real locally pinned LLaVA tokenizer, synthetic visual
embeddings. No vision tower, pretrained decoder, CUDA, or performance claim.
Production persistence, reader, selective engine, and server are exercised.
The independent oracle implements indexed attention without MPIC helpers.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoTokenizer, DynamicCache, LlamaConfig, LlamaForCausalLM
import mmimpress.mpic as mp
from mmimpress.model import LlavaRunner
from mmimpress.piggyback import DecoderVisualHiddenCapture
from mmimpress.store import ChunkReader, IOCounter

SEED = 1234
RTOL, ATOL = 1e-5, 1e-6  # fixed before any measurement
SNAPSHOT = Path("/home/dblab/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def compare(a, b, enforce=False):
    a, b = a.detach().float(), b.detach().float()
    ok = bool(torch.allclose(a, b, rtol=RTOL, atol=ATOL))
    if enforce:
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)
    return {"allclose": ok, "max_abs": float((a - b).abs().max()),
            "mean_abs": float((a - b).abs().mean()), "rtol": RTOL, "atol": ATOL}


class Top(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.config = lm.config
        self.model = SimpleNamespace(language_model=lm.model)
        self.lm_head = lm.lm_head
        self.device = torch.device("cpu")

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def forward(self, **kw):
        return self.lm(**kw)


def expand(tokenizer, prompt, n):
    ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    loc = (ids == 32000).nonzero(as_tuple=True)[0]
    assert len(loc) == 1
    start = int(loc[0])
    return torch.cat((ids[:start], torch.full((n,), 32000), ids[start + 1:])), start


@torch.no_grad()
def native(runner, ids, start, visual):
    embeds = runner.model.get_input_embeddings()(ids[None]).clone()
    embeds[:, start:start + len(visual)] = visual[None]
    pos = torch.arange(len(ids))
    out = runner.model.lm(inputs_embeds=embeds, position_ids=pos[None],
                          cache_position=pos, attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                          use_cache=True, output_hidden_states=True)
    cache = [(x.keys.detach().clone(), x.values.detach().clone()) for x in out.past_key_values.layers]
    return out.logits[:, -1:], cache, embeds, out.hidden_states


def tables(model, positions, like):
    return model.model.rotary_emb(like, torch.tensor(positions)[None])


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    rh = torch.cat((-x[..., half:], x[..., :half]), -1)
    return x * cos[:, None] + rh * sin[:, None]


@torch.no_grad()
def oracle(runner, ids, start, ctx, k):
    """Independent selective decoder; no production plan/mask/assembly/relocation."""
    n, total = ctx.meta["v_token_num"], len(ids)
    k = max(0, min(k, n))
    active = list(range(start + k)) + list(range(start + n, total))
    hidden = runner.model.get_input_embeddings()(ids[active][None]).clone()
    # Direct raw reads, independent of MPICRawReader's range planning.
    dt = np.float16 if ctx.meta["dtype"] == "float16" else np.float32
    visual = torch.from_numpy(np.fromfile(ctx.dir / "visual_input.bin", dtype=dt).copy()).reshape(n, -1).float()
    if k:
        hidden[:, start:start + k] = visual[:k][None]
    source_pos = list(range(ctx.meta["v_token_start"], ctx.meta["v_token_start"] + n))
    target_pos = list(range(start, start + n))
    results, hidden_layers = [], []
    for li, layer in enumerate(runner.layers):
        def read(kind):
            a = torch.from_numpy(np.fromfile(ctx.dir / f"layer_{li:02d}/{kind}.bin", dtype=dt).copy())
            return a.reshape(n, ctx.meta["num_heads"], ctx.meta["head_dim"]).permute(1, 0, 2)[None].float()
        ik, iv = read("k"), read("v")
        if source_pos != target_pos:
            sc, ss = tables(runner.model.lm, source_pos, ik)
            tc, ts = tables(runner.model.lm, target_pos, ik)
            # Independently undo the old phase, then apply the new phase.
            ik = rotate(rotate(ik, sc, -ss) / (sc[:, None] ** 2 + ss[:, None] ** 2), tc, ts)
        residual = hidden
        normed = layer.input_layernorm(hidden)
        attn = layer.self_attn
        def project(mod):
            return mod(normed).reshape(1, len(active), -1, attn.head_dim).transpose(1, 2)
        q, ak, av = project(attn.q_proj), project(attn.k_proj), project(attn.v_proj)
        c, s = tables(runner.model.lm, active, normed)
        q, ak = rotate(q, c, s), rotate(ak, c, s)
        fk = torch.empty(1, ik.shape[1], total, ik.shape[3])
        fv = torch.empty_like(fk)
        fk[:, :, start:start + n] = ik
        fv[:, :, start:start + n] = iv
        for ai, position in enumerate(active):
            fk[:, :, position] = ak[:, :, ai]
            fv[:, :, position] = av[:, :, ai]
        kq = fk.repeat_interleave(attn.num_key_value_groups, dim=1)
        vq = fv.repeat_interleave(attn.num_key_value_groups, dim=1)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, kq) * attn.scaling
        allowed = torch.arange(total)[None] <= torch.tensor(active)[:, None]
        probs = scores.masked_fill(~allowed[None, None], float("-inf")).softmax(-1)
        attended = torch.einsum("bhqk,bhkd->bhqd", probs, vq).transpose(1, 2).reshape(1, len(active), -1)
        hidden = residual + attn.o_proj(attended)
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        hidden_layers.append(hidden.clone())
        results.append((fk, fv))
    logits = runner.model.lm_head(runner.model.lm.model.norm(hidden)[:, -1:])
    return logits, results, hidden_layers


class Observer:
    """Read-only tensor hooks plus wrappers restored on context exit."""
    def __init__(self, runner, case):
        self.runner, self.case = runner, case
        self.layers = [dict(layer=i, projections={}, projection_inputs={}) for i in range(len(runner.layers))]
        self.handles, self.reads, self.timeline, self.positions = [], [], [], []
        self.mask, self.post, self.hidden = None, {}, {}
        self.li, self.product, self.t0 = -1, 0, time.perf_counter()

    def __enter__(self):
        for li, layer in enumerate(self.runner.layers):
            for kind in ("q", "k", "v"):
                def capture(module, args, output, li=li, kind=kind):
                    self.layers[li]["projection_inputs"][kind] = list(args[0].shape)
                    self.layers[li]["projections"][kind] = list(output.shape)
                    self.layers[li].setdefault("projection_calls", {}).setdefault(kind, 0)
                    self.layers[li]["projection_calls"][kind] += 1
                    if kind == "q":
                        self.li, self.product = li, 0
                        self.timeline.append({"event": "compute_q", "layer": li, "at_s": time.perf_counter() - self.t0})
                self.handles.append(getattr(layer.self_attn, kind + "_proj").register_forward_hook(capture))
            def post(module, args, li=li):
                self.post[li] = args[0].detach().clone()
            def mlp(module, args, output, li=li):
                self.hidden[li] = self.post[li] + output.detach()
                self.layers[li]["mlp_input_shape"] = list(args[0].shape)
                self.layers[li]["hidden_to_next_layer_shape"] = list(self.hidden[li].shape)
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(post))
            self.handles.append(layer.mlp.register_forward_hook(mlp))
        def rotary(module, args, output):
            self.positions.append(args[1].detach().cpu().tolist())
        self.handles.append(self.runner.model.lm.model.rotary_emb.register_forward_hook(rotary))
        self.orig_assemble, self.orig_mask = mp.assemble_linked_kv, mp.build_causal_mask
        self.orig_matmul, self.orig_pread = torch.matmul, os.pread
        def mask(*args, **kw):
            self.mask = self.orig_mask(*args, **kw)
            self.active = torch.as_tensor(args[0]).cpu()
            return self.mask
        def assemble(ik, iv, ak, av, positions, start, total, dummy_value=0):
            fk, fv = self.orig_assemble(ik, iv, ak, av, positions, start, total, dummy_value)
            d = self.layers[self.li]
            d.update(active_K_shape=list(ak.shape), active_V_shape=list(av.shape),
                     full_K_shape=list(fk.shape), full_V_shape=list(fv.shape),
                     active_positions=positions.cpu().tolist(), replaced_K_rows=len(positions), replaced_V_rows=len(positions),
                     active_K_installed=bool(torch.equal(fk[:, :, positions], ak)),
                     active_V_installed=bool(torch.equal(fv[:, :, positions], av)))
            reused = sorted(set(range(start, start + ik.shape[-2])) - set(positions.tolist()))
            local = [x - start for x in reused]
            d.update(reused_rows=len(reused), unchanged_reused_K=bool(torch.equal(fk[:, :, reused], ik[:, :, local])),
                     unchanged_reused_V=bool(torch.equal(fv[:, :, reused], iv[:, :, local])))
            expected = torch.arange(total)[None] <= positions.cpu()[:, None]
            actual = self.mask[0, 0].cpu() == 0
            d.update(causal_mask_shape=list(self.mask.shape), original_position_causal_mask_correct=bool(torch.equal(expected, actual)),
                     future_allowed_count=int((actual & ~expected).sum()), missing_past_count=int((expected & ~actual).sum()),
                     suffix_query_sees_all_image_keys=bool(actual[-1, start:start + ik.shape[-2]].all()),
                     installed_before_QK=True)
            self.timeline.append({"event": "install_KV", "layer": self.li, "at_s": time.perf_counter() - self.t0})
            return fk, fv
        def matmul(a, b, *args, **kw):
            result = self.orig_matmul(a, b, *args, **kw)
            if self.li >= 0 and a.ndim == 4 and b.ndim == 4:
                d = self.layers[self.li]
                if self.product == 0:
                    d.update(Q_shape=list(a.shape), attention_K_transposed_shape=list(b.shape), QK_shape=list(result.shape))
                    self.timeline.append({"event": "QK", "layer": self.li, "at_s": time.perf_counter() - self.t0})
                elif self.product == 1:
                    d.update(attention_V_shape=list(b.shape), attention_output_shape=list(result.shape),
                             future_probability_max=float(a.masked_select(self.mask.expand_as(a) != 0).abs().max()) if bool((self.mask != 0).any()) else 0.)
                self.product += 1
            return result
        def pread(fd, count, offset):
            path = os.readlink(f"/proc/self/fd/{fd}")
            t = time.perf_counter()
            result = self.orig_pread(fd, count, offset)
            self.reads.append({"path": path, "requested": count, "returned": len(result), "offset": offset})
            self.timeline.append({"event": "pread", "path": Path(path).name, "start_s": t - self.t0,
                                  "end_s": time.perf_counter() - self.t0, "thread": "main", "bytes": len(result)})
            return result
        mp.assemble_linked_kv, mp.build_causal_mask = assemble, mask
        torch.matmul, os.pread = matmul, pread
        return self

    def __exit__(self, *exc):
        mp.assemble_linked_kv, mp.build_causal_mask = self.orig_assemble, self.orig_mask
        torch.matmul, os.pread = self.orig_matmul, self.orig_pread
        for handle in self.handles:
            handle.remove()


@contextlib.contextmanager
def cpu_legacy_sync():
    # Legacy Server hard-codes CUDA sync. Only this CPU diagnostic substitutes
    # a no-op; no GPU timing or production code is changed.
    original = torch.cuda.synchronize
    torch.cuda.synchronize = lambda *a, **kw: None
    try:
        yield
    finally:
        torch.cuda.synchronize = original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    cfg = LlamaConfig(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=2048, attention_dropout=0., attn_implementation="eager")
    cfg.image_token_index = 32000
    lm = LlamaForCausalLM(cfg).float().eval()
    runner = SimpleNamespace(model=Top(lm), layers=lm.model.layers, model_id="audit/random-llama",
                             processor=SimpleNamespace(tokenizer=tokenizer, image_processor=None),
                             load_4bit=False, image_token_id=32000)
    runner.prompt = lambda q: LlavaRunner.prompt(runner, q)
    runner.visual_spans = lambda ids: LlavaRunner.visual_spans(runner, ids)
    runner.visual_span = lambda ids: LlavaRunner.visual_span(runner, ids)
    # Synthetic structure annotations, not a reproduction of AnyRes geometry.
    runner.anyres_layout = lambda size, n: (8, 3, 7, [63, 79, 95])
    n = 96
    visual = torch.randn(n, cfg.hidden_size).half().float()
    source_prompt = runner.prompt("What is shown?")
    source_ids, source_start = expand(tokenizer, source_prompt, n)
    capture = DecoderVisualHiddenCapture(runner, source_start, n)
    with capture:
        ref, source_cache, _, _ = native(runner, source_ids, source_start, visual)
    production_store = out / "production_fp16"
    persisted = mp.persist_captured_mpic_prefix(runner, source_cache, source_ids, [8, 8], capture.result_cpu(),
        production_store, image_id="synthetic", hidden_capture_stats=capture, chunk_size=64,
        image_input_sha256=hashlib.sha256(visual.numpy().tobytes()).hexdigest())
    context = mp.MPICContext(production_store, "cpu", drop_cache=False, runner=runner)
    dump(out / "persistence.json", persisted)
    diagnostic_store = out / "diagnostic_fp32"
    diagnostic_store.mkdir()
    (diagnostic_store / "visual_input.bin").write_bytes(visual.numpy().tobytes())
    for li, pair in enumerate(source_cache):
        folder = diagnostic_store / f"layer_{li:02d}"
        folder.mkdir()
        for kind, tensor in zip(("k", "v"), pair):
            (folder / f"{kind}.bin").write_bytes(tensor[0, :, source_start:source_start + n].permute(1, 0, 2).contiguous().numpy().tobytes())
    dm = dict(context.meta, dtype="float32", bytes_visual_kv=3 * 2 * n * 4 * 16 * 4)
    diagnostic = SimpleNamespace(dir=diagnostic_store, meta=dm, reader=mp.MPICRawReader(diagnostic_store, dm, drop_cache=False))
    before = {str(p.relative_to(out)): sha(p) for p in out.glob("*/*") if p.is_file()}
    # Record every payload, including interior bytes, rather than echoing stats.
    payload_before = {str(p.relative_to(out)): sha(p) for p in out.glob("*/*/*.bin")}
    payload_before.update({str(p.relative_to(out)): sha(p) for p in out.glob("*/visual_input.bin")})
    dump(out / "config.json", {"seed": SEED, "RTOL": RTOL, "ATOL": ATOL, "cpu_threads": 2,
        "random_decoder_config": cfg.to_dict(), "tokenizer_snapshot": str(SNAPSHOT),
        "visual_tokens": n, "synthetic_structural_local_rows": [63, 79, 95],
        "pretrained_model_weights_loaded": False, "real_image_pixels_processed": False,
        "quantization": None, "production_store_dtype": "float16", "diagnostic_store_dtype": "float32",
        "timing_scope": "CPU mechanism diagnostics only; not comparable with published GPU TTFT"})
    traces, results = [], []

    def probe(case, prompt, k, ctx, enforce_dense=False, dummy=0.):
        ids, start = expand(tokenizer, prompt, n)
        io = IOCounter()
        with Observer(runner, case) as obs:
            logits, cache, plan, stats = mp.MPICSelectivePrefill(runner, ctx, k, dummy).run(ids, start, io)
        ol, oc, oh = oracle(runner, ids, start, ctx, k)
        dense, dc, _, _ = native(runner, ids, start, visual)
        checks = {"logits_vs_independent_selective": compare(logits, ol, True),
                  "logits_vs_dense_ReComp": compare(logits, dense, enforce_dense),
                  "first_token_same_as_dense": int(logits.argmax(-1)) == int(dense.argmax(-1))}
        for li, ((ak, av), (ek, ev), d) in enumerate(zip(cache, oc, obs.layers)):
            d["K_vs_independent_selective"] = compare(ak, ek, True)
            d["V_vs_independent_selective"] = compare(av, ev, True)
            d["hidden_vs_independent_selective"] = compare(obs.hidden[li], oh[li], True)
            d["K_vs_dense"] = compare(ak, dc[li][0], enforce_dense)
            d["V_vs_dense"] = compare(av, dc[li][1], enforce_dense)
            assert all(d[x] for x in ("active_K_installed", "active_V_installed", "unchanged_reused_K", "unchanged_reused_V", "original_position_causal_mask_correct"))
            assert d["projection_calls"] == {"q": 1, "k": 1, "v": 1}
            assert d["Q_shape"][2] == plan.n_active
            assert d["future_probability_max"] == 0
        assert sum(x["returned"] for x in obs.reads) == io.summary()["bytes"]
        assert len(obs.reads) == io.summary()["preads"]
        types = [{"position": i, "id": int(t), "token": tokenizer.convert_ids_to_tokens(int(t)),
                  "kind": "image_structure_synthetic" if start <= i < start + n and i - start in [63, 79, 95]
                  else "image_visual" if start <= i < start + n else "text_or_template"} for i, t in enumerate(ids)]
        trace = {"case": case, "store_dtype": ctx.meta["dtype"], "k": k, "prompt": prompt,
                 "tokens": types, "original_positions": list(range(len(ids))),
                 "selected_image_local_rows": list(range(plan.k_recompute)),
                 "active_positions": plan.active_positions.tolist(), "layers": obs.layers,
                 "rotary_position_calls": obs.positions, "preads": obs.reads, "timeline": obs.timeline,
                 "stats": stats, "io": io.summary(), "checks": checks}
        traces.append(trace)
        results.append({"case": case, "status": "PASS", "k": k, "dtype": ctx.meta["dtype"], "total": len(ids),
                        "active": plan.n_active, "reused_image": n - plan.k_recompute, "preads": io.summary()["preads"],
                        "bytes": io.summary()["bytes"], "checks": checks})
        return logits, cache

    # Warm-up is separately labelled and excluded from measurement summary.
    warm = mp.MPICServer(runner, max_new_tokens=3).request(context, question="Warm-up fixture", cold=False)
    dump(out / "warmup.json", warm)
    prompt = runner.prompt("Describe the object in a short phrase.")
    for k in (0, 32, n):
        probe(f"single_image_fp32_k{k}", prompt, k, diagnostic, enforce_dense=True)
        probe(f"single_image_production_fp16_k{k}", prompt, k, context, enforce_dense=k == n)
    base_logits, base_cache = probe("dummy_zero", prompt, 32, diagnostic)
    sentinel_logits, sentinel_cache = probe("dummy_sentinel", prompt, 32, diagnostic, dummy=12345.)
    assert torch.equal(base_logits, sentinel_logits)
    assert all(torch.equal(a, b) for ap, bp in zip(base_cache, sentinel_cache) for a, b in zip(ap, bp))
    # Same token count before the image; then a real position shift.
    changed = "NOTE: <image>\nDescribe the object in a short phrase. ASSISTANT:"
    shifted = "A conversation between a helpful assistant and a curious user. USER: <image>\nDescribe the object in a short phrase. ASSISTANT:"
    _, changed_start = expand(tokenizer, changed, n)
    assert changed_start == source_start
    for name, text in (("changed_pre_image_same_position", changed), ("changed_pre_image_shifted", shifted)):
        probe(name + "_k32", text, 32, context)
        probe(name + "_kN", text, n, context, enforce_dense=True)
    # One-token suffix boundary: the legacy FullLoad layer hook mistakes this
    # legitimate prefill for decode, which can leave all visual K/V at zero.
    from mmimpress.serve import PrefixCache, Server
    full_meta = dict(dm, prefix_len=source_start + n, n_chunks_per_layer=2, newline_idx=[63, 79, 95], n_spatial=n - 3)
    syskv = {kind: torch.stack([pair[ki][0, :, :source_start] for pair in source_cache]) for ki, kind in enumerate(("k", "v"))}
    fullctx = SimpleNamespace(meta=full_meta, reader=ChunkReader(diagnostic_store, full_meta, drop_cache=False),
                              cache=PrefixCache(full_meta, syskv, "cpu", dtype=torch.float32),
                              separator_positions=lambda li: [63, 79, 95])
    with cpu_legacy_sync():
        suffix_ids = expand(tokenizer, prompt, n)[0][source_start + n:]
        # Observe first-token logits without changing their return values.
        recorded = []
        h = lm.lm_head.register_forward_hook(lambda mod, args, val: recorded.append(val.detach().clone()))
        try:
            full = Server(runner, max_new_tokens=3).request(fullctx, mode="fullload", suffix_ids=suffix_ids, cold=False)
        finally:
            h.remove()
        k0, _ = probe("full_load_contract_fp32_k0", prompt, 0, diagnostic, enforce_dense=True)
        full_agreement = compare(recorded[0][:, -1:], k0, True)
        edge_recorded = []
        h = lm.lm_head.register_forward_hook(lambda mod, args, val: edge_recorded.append(val.detach().clone()))
        try:
            edge = Server(runner, max_new_tokens=1).request(fullctx, mode="fullload", suffix_ids=torch.tensor([12]), cold=False)
        finally:
            h.remove()
        edge_image_nonzero = [{"K": int(torch.count_nonzero(k[:, :, source_start:source_start + n])),
                               "V": int(torch.count_nonzero(v[:, :, source_start:source_start + n]))}
                              for k, v in zip(fullctx.cache.k, fullctx.cache.v)]
        edge_ids = torch.cat((source_ids[:source_start + n], torch.tensor([12])))
        edge_expected, _, _, _ = native(runner, edge_ids, source_start, visual)
        edge_logits = compare(edge_recorded[0][:, -1:], edge_expected)
        assert all(v == {"K": 0, "V": 0} for v in edge_image_nonzero)
        # Incoming pre-image text is discarded by legacy FullLoad's public
        # prompt helper; record this as a contract/validation limitation.
        changed_contract = prompt.replace("USER: ", "NOTE: ", 1)
        prefix_recorded = []
        h = lm.lm_head.register_forward_hook(lambda mod, args, val: prefix_recorded.append(val.detach().clone()))
        try:
            old_full = Server(runner, max_new_tokens=1).request(fullctx, mode="fullload", prompt_text=prompt, cold=False)
            changed_full = Server(runner, max_new_tokens=1).request(fullctx, mode="fullload", prompt_text=changed_contract, cold=False)
        finally:
            h.remove()
        assert torch.equal(prefix_recorded[0], prefix_recorded[1])
    dump(out / "fullload_contract.json", {"normal_suffix_logits_vs_k0": full_agreement,
        "normal": full, "single_token_suffix": edge,
        "single_token_suffix_issue": "FAIL" if edge["io"]["bytes"] == 0 else "PASS",
        "single_token_image_installed_nonzero_elements_per_layer": edge_image_nonzero,
        "single_token_logits_vs_correct_context": edge_logits,
        "changed_pre_image_input_silently_ignored": True,
        "changed_prefix_contract_scope": "outside FullLoad fixed-prefix contract; no validation rejects it",
        "CPU_shim": "CUDA sync no-op; PrefixCache dtype FP32; diagnostic FP32 source; no GPU timing claim"})
    # Three-turn histories are generated separately for all three methods.
    spec = importlib.util.spec_from_file_location("audit_main", ROOT / "scripts/108_eval_llava_mt_gqa_B.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dialogue = {"dialog_id": "cpu_synthetic", "turns": [
        {"turn_id": 1, "question_id": "a1", "question": "What is shown?"},
        {"turn_id": 2, "question_id": "a2", "question": "What color is it?"},
        {"turn_id": 3, "question_id": "a3", "question": "Repeat the main object name."}]}
    histories, generations = {}, []
    for method in ("recompute", "fullload", "mpic32_ssd"):
        histories[method] = {}
        for turn in (1, 2, 3):
            text, history, entries = module.prompt_factory(dialogue, turn, method, histories[method])
            ids, start = expand(tokenizer, text, n)
            # Actual normal T1 uses pixels; CPU T1 here uses synthetic visual
            # embeddings through a native full decoder traversal for each method.
            if turn == 1 or method == "recompute":
                _, _, embeds, _ = native(runner, ids, start, visual)
                with torch.no_grad():
                    sequence = lm.generate(inputs_embeds=embeds, attention_mask=torch.ones(1, len(ids), dtype=torch.long), max_new_tokens=3, do_sample=False,
                                           pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
                answer = tokenizer.decode(sequence[0].tolist(), skip_special_tokens=True).strip()
                token_ids = sequence[0].tolist()
            elif method == "mpic32_ssd":
                probe(f"generated_history_t{turn}_mpic32", text, 32, context)
                response = mp.MPICServer(runner, max_new_tokens=3).request(context, prompt_text=text, cold=False)
                answer, token_ids = response["prediction"], response["generated_token_ids"]
                assert response["decode_cache_append_exact"]
            else:
                with cpu_legacy_sync():
                    response = Server(runner, max_new_tokens=3).request(fullctx, mode="fullload", suffix_ids=ids[start + n:], cold=False)
                answer, token_ids = response["answer"], response["generated_token_ids"]
            row = dict(method_id=method, dialog_id=dialogue["dialog_id"], turn_id=turn, prediction=answer,
                       logical_request_id=f"cpu:{method}:t{turn}", physical_execution_id=f"cpu:{method}:t{turn}")
            generations.append(dict(row, prompt=text, history=history, history_entries=entries, generated_token_ids=token_ids))
            histories[method][turn] = row
    dump(out / "generated_history.json", generations)
    rejected = []
    for case, text in (("interleaved_two_images", "Text A <image> Text B <image> Text C"),
                       ("mixed_hit_miss_two_images", "Cached image <image> Missing image <image> Compare them.")):
        original = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        image_slots = [i for i, token in enumerate(original) if token == 32000]
        # Show both hypothetical global-k and per-image-k sets; neither has
        # an executable production selector. No oracle choice is attributed to MPIC.
        expected_starts = [image_slots[0], image_slots[1] + n - 1]
        io_before = len(traces)
        with Observer(runner, case) as obs:
            try:
                mp.MPICServer(runner, max_new_tokens=1).request(context, prompt_text=text, cold=False)
            except ValueError as e:
                error = str(e)
            else:
                raise AssertionError("two-image input unexpectedly accepted")
        assert not obs.reads and not any(x["projections"] for x in obs.layers)
        rejected.append({"case": case, "status": "FAIL", "scope": "unsupported paper workload",
                         "error": error, "unexpanded_input_ids": original, "image_placeholder_positions": image_slots,
                         "hypothetical_expanded_image_starts": expected_starts,
                         "global_k32_if_defined": list(range(expected_starts[0], expected_starts[0] + 32)),
                         "per_image_k32_if_defined": [list(range(s, s + 32)) for s in expected_starts],
                         "preads": 0, "decoder_projection_calls": 0,
                         "layer_shapes": "NOT APPLICABLE: rejected before forward", "fallback": False})
    dump(out / "multi_image_rejections.json", rejected)
    # A real file short-read must propagate, never return a ReComp answer.
    fault_dir = out / "fault_store"
    import shutil
    shutil.copytree(production_store, fault_dir)
    fault = mp.MPICContext(fault_dir, "cpu", drop_cache=False, runner=runner)
    path = fault_dir / "layer_00/v.bin"
    with path.open("r+b") as f:
        f.truncate(5)
    try:
        mp.MPICServer(runner, max_new_tokens=1).request(fault, prompt_text=prompt, cold=False)
    except OSError as exc:
        dump(out / "failure_propagation.json", {"status": "PASS", "error": str(exc), "fallback": False,
                                                 "fault_only_in_new_audit_store": True})
    else:
        raise AssertionError("file short read was hidden")
    fault.close()
    # Separate best-effort DONTNEED check; physical NAND/controller residency
    # is deliberately not inferred from the syscall's success.
    response = mp.MPICServer(runner, max_new_tokens=3).request(context, prompt_text=prompt, cold=True)
    dump(out / "cpu_request_timing.json", {"response": response,
        "physical_cold_ssd": "NOT VERIFIED", "GPU_H2D": "NOT APPLICABLE (CPU)",
        "resident_metadata_bytes": context.metadata_resident_bytes,
        "OS_page_cache_method": "posix_fadvise_DONTNEED advisory",
        "inclusive_prefill_interval_must_not_be_added_to_its_IO_and_assembly_subintervals": True})
    payload_after = {name: sha(out / name) for name in payload_before}
    assert payload_before == payload_after
    dump(out / "source_payload_integrity.json", {"status": "PASS", "before": payload_before, "after": payload_after,
                                               "fault_store_excluded": True})
    with (out / "layer_traces.jsonl").open("x") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")
    dump(out / "summary.json", {"mechanism_cases": results, "dummy_invariance": "PASS",
        "full_payload_immutable": "PASS", "multi_image": rejected,
        "generated_history_requests": len(generations), "no_silent_fallback": "PASS",
        "legacy_fullload_single_suffix": "FAIL" if edge["io"]["bytes"] == 0 else "PASS",
        "GPU_pretrained_quantized_model": "NOT VERIFIED", "mixed_hit_miss_overlap": "FAIL (unsupported)",
        "claim": "CPU selective decoder correctness only; not pretrained MLLM quality or GPU timing"})
    context.close()
    diagnostic.reader.close()
    fullctx.reader.close()
    print(json.dumps({"mechanism_cases_passed": len(results), "generated_history_requests": len(generations),
                      "multi_image_rejections": len(rejected), "legacy_fullload_single_suffix": edge["io"]["bytes"],
                      "GPU": "NOT VERIFIED"}))


if __name__ == "__main__":
    main()
