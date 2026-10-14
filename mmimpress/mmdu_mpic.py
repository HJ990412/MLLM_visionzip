"""Distinct prefix-caching and two-pass full-reuse comparators.

Full reuse here is intentionally NOT the fixed-prefix legacy FullLoad.
Native Llama text prefill omits images, links independent image KV, and then
re-forwards the last text token with that linked cache to produce first logits.
This is a documented Transformers adaptation of the paper's two-step process.
"""
from dataclasses import dataclass
import torch
from mmimpress.mpic import relocate_post_rope_keys, seed_dynamic_cache
from mmimpress.mpic_multi import assemble_multi_kv


@dataclass
class PrefixState:
    labels: list
    cache: list
    logits: torch.Tensor
    embeds: torch.Tensor


def matching_prefix_length(labels, previous):
    matched = 0
    if previous is not None:
        for a, b in zip(labels, previous.labels):
            if a != b:
                break
            matched += 1
    return matched


def prompt_from_generated(dialog, turn, generated):
    """Keep every marker in place; never substitute a gold answer."""
    if set(generated) != set(range(1, turn)):
        raise ValueError("missing/future generated history")
    parts = []
    for t in dialog["turns"][:turn]:
        text = t["question"].replace("<ImageHere>", "<image>")
        value = f"USER: {text} ASSISTANT:"
        if t["turn_id"] < turn:
            value += " " + generated[t["turn_id"]] + "</s>"
        parts.append(value)
    return " ".join(parts)


def logical_labels(ids, bindings):
    labels = [(int(token), None) for token in ids]
    for b in bindings:
        # image_id names the immutable image content; external benchmark
        # preparation hashes each source image and verifies it before execution.
        for local in range(b.count):
            labels[b.start + local] = (int(ids[b.start + local]), (b.image_id, local))
    return labels


@torch.no_grad()
def prefix_prefill(runner, ids, embeds, labels, previous=None):
    matched = matching_prefix_length(labels, previous)
    if matched == len(labels) and previous is not None:
        return previous.logits, previous.cache, previous, matched
    model, device = runner.model, torch.device(runner.model.device)
    cache = None
    if matched:
        cache = seed_dynamic_cache([(k[:, :, :matched].clone(), v[:, :, :matched].clone())
                                    for k, v in previous.cache], config=model.config)
    pos = torch.arange(matched, len(ids), device=device)
    output = model.model.language_model(inputs_embeds=embeds[:, matched:],
        position_ids=pos[None], cache_position=pos,
        attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=device),
        past_key_values=cache, use_cache=True)
    layers = [(x.keys, x.values) for x in output.past_key_values.layers]
    logits = model.lm_head(output.last_hidden_state[:, -1:])
    state = PrefixState(list(labels), layers, logits, embeds)
    return logits, layers, state, matched


@torch.no_grad()
def full_reuse_prefill(runner, ids, bindings, counter):
    model, device = runner.model, torch.device(runner.model.device)
    lm = model.model.language_model
    image = torch.zeros(len(ids), dtype=torch.bool)
    for b in bindings:
        image[b.start:b.start + b.count] = True
    text = (~image).nonzero(as_tuple=True)[0].to(device)
    if not len(text) or int(text[-1]) != len(ids) - 1:
        raise ValueError("full reuse requires a final text token")
    embeddings = model.get_input_embeddings()(ids[text.cpu()].to(device)[None])
    # Text-only cache indices are compact, but RoPE uses original positions.
    # Using original cache_position here would erroneously expose future text.
    compact = torch.arange(len(text), device=device)
    text_output = lm(inputs_embeds=embeddings, position_ids=text[None],
        cache_position=compact, attention_mask=torch.ones(1, len(text), dtype=torch.long, device=device),
        use_cache=True)
    layers = []
    for li, layer in enumerate(text_output.past_key_values.layers):
        pairs = []
        for b in bindings:
            ctx = b.context
            rk, ck = ctx.reader.read_reused_layer(li, "k", 0, counter)
            rv, cv = ctx.reader.read_reused_layer(li, "v", 0, counter)
            if not torch.equal(rk, rv) or rk.tolist() != list(range(b.count)):
                raise AssertionError("full reuse did not load the complete image")
            ik = ck.to(device, embeddings.dtype).permute(1, 0, 2)[None]
            iv = cv.to(device, embeddings.dtype).permute(1, 0, 2)[None]
            source = int(ctx.meta["v_token_start"])
            if source != b.start:
                sp, tp = [torch.arange(s, s + b.count, device=device)[None] for s in (source, b.start)]
                sc, ss = lm.rotary_emb(ik, sp)
                tc, ts = lm.rotary_emb(ik, tp)
                ik = relocate_post_rope_keys(ik, sc, ss, tc, ts)
            pairs.append((ik, iv))
        fk, fv = assemble_multi_kv(pairs, layer.keys, layer.values, text,
                                   [(b.start, b.count) for b in bindings], len(ids))
        # Avoid appending a duplicate last token in the second pass.
        layers.append((fk[:, :, :-1].contiguous(), fv[:, :, :-1].contiguous()))
    cache = seed_dynamic_cache(layers, config=model.config)
    last = torch.tensor([len(ids) - 1], device=device)
    out = model(input_ids=ids[-1:].to(device)[None],
        attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=device),
        position_ids=last[None], cache_position=last, past_key_values=cache, use_cache=True)
    complete = [(x.keys, x.values) for x in cache.layers]
    return out.logits[:, -1:], complete, {"decoder_prefill_pass_count": 2,
        "text_prefill_rows": len(text), "linked_first_token_rows": 1,
        "image_rows_retained": sum(b.count for b in bindings), "fallback": False,
        "implementation_difference": "last text token re-forward after text-only prefill and linking"}
