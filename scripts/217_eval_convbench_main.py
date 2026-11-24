#!/usr/bin/env python3
"""ConvBench main experiment: ReComp / FullLoad / IMPRESS / Ours on LLaVA-NeXT-7b and LLaVA-1.5-7b
(contract docs/convbench_main_contract.md).

No method code is changed or added.  Where the code comes from:

* LLaVA-NeXT: scripts/209_eval_llava_next_mtvqa_main.py is imported and its request, capture, writer and
  activation functions are called as they are.  They are the FullLoad / IMPRESS calls of the 2026-11-17 run
  (scripts/196, results/mt_gqa_impress_code_20261117T185832Z) and the "Ours(원래 순서)" calls of the
  2026-11-20 run (scripts/202, results/mt_gqa_ours_order_20261120T052725Z: PYR-beta-raw, 25 %,
  ``row_order="original"``).
* LLaVA-1.5: scripts/206_eval_llava15_mt_gqa_main.py (results/llava15_mt_gqa_main_20261120T165001Z) is
  imported and its functions are called as they are (FullLoad, IMPRESS, Ours = PYR-b20, original row order).
* ConvBench settings: docs/convbench_baseline_alignment.md.  The prompt renderer, the official-token-id
  check and the cap (always 1024) are the functions / constants of scripts/213_convbench_recomp_generate.py
  (imported).

What this script adds is the ConvBench request loop: the prompt (system sentence before the image, ``</s>``
between turns, official token ids), ``max_new_tokens = 1024`` on the server object, one row per request,
the conversation commit / resume logic and the failure record.

One conversation at a time: every arm computes turn 1 from pixels; the arm that owns a store kind writes
that store from its own turn 1; turns 2-3 of the stored arms read it from the cold SSD (page cache dropped
before each request); the conversation is committed; its stores are deleted.

A request that raises is recorded (method, turn, cause).  The first time that happens in a conversation, the
conversation is started again from turn 1 for every arm; if it happens again the conversation is committed
with the failed request(s) recorded and the later turns of that arm marked as failed.  Denominators stay 577.

Resume: committed conversations are skipped; a conversation that was interrupted starts again at turn 1 for
every arm, its partial rows are moved to ``discarded/`` (never read) and its stores are removed.

Stages: --stage freeze | prerun            (no model)
        --stage gate | pilot | overflow | main  --model next | llava15
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import convbench_main_common as C  # noqa: E402
import llava_next_main_status as STATUS  # noqa: E402  (generic status.json / SSD bookkeeping helpers)

GiB = 1 << 30
GPU_WAIT_SECONDS = 6 * 3600
FREE_SPACE_FACTOR = 20
# rows keep these result fields only outside the gate phase (the gate phase keeps everything)
DROP_ALWAYS = ("captured_past_key_values", "first_logits", "logits", "answer", "generated_token_ids")
DROP_LARGE = ("layer_log", "probe_sources", "layer_diagnostics", "raster_chunks", "importance_only_chunks", "os_read_trace",
              "conditioned_files", "projection_calls", "selected_original_ids", "layer_keep_ratios", "k_per_layer",
              "attended_content_kv_count_per_layer", "keep_count_per_layer", "planned_normal_read_bytes_per_layer",
              "planned_packed_read", "logical_content_retention_ratio_per_layer", "normal_chunks_read_per_layer")
ALIGN_SECONDS_PER_CONVERSATION = {"next": 7626.8 / 577, "llava15": 4664.1 / 577}      # ReComp only, alignment run (measured)
ALIGN_JUDGE_SECONDS_PER_ANSWER_SET = 12911.0                                           # 577 x 3 judgments, alignment run
# the ConvBench prompt has the system sentence before the image (30 more prefix tokens than the MT-GQA prompt): every store's
# system-prefix file is this much larger than the store-size formulas of the reference runs assume (measured, rehearsal 2026-11-22)
CONVBENCH_SYSTEM_PREFIX_EXTRA_BYTES = 15_703_472


class ConversationRetry(Exception):
    """A request failed in the first attempt of a conversation: the conversation starts again."""


class FatalState(Exception):
    """The process is not in a state that can serve another request (a hook or a patch was left behind)."""


def load_script(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def gen_module():
    """scripts/213 (alignment generator): prompt renderer, official token ids, render table."""
    return load_script("_convbench_main_alignment_gen", "213_convbench_recomp_generate.py")


# ================================================================== backends
class Backend:
    """The imported reference harness of one model and the mapping role -> its arm id."""

    def __init__(self, model):
        self.model = model
        if model == "next":
            B = load_script("_convbench_main_next_backend", "209_eval_llava_next_mtvqa_main.py")
            self.arms = {C.RECOMP: B.RECOMP, C.FULL: B.FULL_ASYNC, C.IMP_SYNC: B.NEW, C.IMP: B.NEW_ASYNC, C.IMP_R100: B.NEW_R100,
                         C.FULL_ON_IMP: B.FULL_REORDERED, C.OURS_SYNC: B.ORIG, C.OURS: B.ORIG_ASYNC, C.OURS_REPEAT: B.ORIG_REPEAT}
            self.reference = {"fullload_impress": "results/mt_gqa_impress_code_20261117T185832Z",
                              "ours": "results/mt_gqa_ours_order_20261120T052725Z",
                              "harness": "scripts/209_eval_llava_next_mtvqa_main.py"}
            self.vector_source = B.VECTOR_SOURCE
        else:
            B = load_script("_convbench_main_llava15_backend", "206_eval_llava15_mt_gqa_main.py")
            self.arms = {C.RECOMP: "recompute", C.FULL: "fullload_async", C.IMP_SYNC: B.NEW, C.IMP: B.NEW_ASYNC, C.IMP_R100: B.NEW_R100,
                         C.FULL_ON_IMP: B.FULL_REORDERED, C.OURS_SYNC: B.OURS, C.OURS: B.OURS_ASYNC, C.OURS_REPEAT: B.OURS_REPEAT}
            self.reference = {"fullload_impress": "results/llava15_mt_gqa_main_20261120T165001Z",
                              "ours": "results/llava15_mt_gqa_main_20261120T165001Z",
                              "harness": "scripts/206_eval_llava15_mt_gqa_main.py"}
            self.vector_source = B.VECTOR_SOURCE
        # dataset identity written into the store metadata by the imported writer calls
        B.DATASET = "ConvBench"
        B.SCHEMA = C.SCHEMA
        self.B = B
        self.runner = self.servers = None
        self._docs = None

    # ---- static description (no model needed)
    def kind(self, role):
        return self.B.ARMS[self.arms[role]]["kind"]

    def family(self, role):
        return self.B.ARMS[self.arms[role]]["family"]

    def describe(self):
        B = self.B
        return {"model_id": B.MODEL_ID, "revision": B.REVISION, "reference": self.reference,
                "arms": {role: {"backend_arm": arm, **{k: v for k, v in B.ARMS[arm].items() if k != "display"}} for role, arm in self.arms.items()},
                "server_args": B.SERVER_ARGS, "impress": B.OFFICIAL, "readahead": {k: v for k, v in B.READAHEAD.items()},
                "vector": {"id": B.VECTOR, "file": str(Path(self.vector_source).relative_to(ROOT)), "file_sha256": B.VECTOR_FILE_SHA,
                           "ratio_sha256": B.VECTOR_RATIO_SHA},
                "store_kinds": {role: self.kind(role) for role in self.arms}}

    def docs(self):
        if self._docs is None:
            B = self.B
            assert C.sha(self.vector_source) == B.VECTOR_FILE_SHA
            if self.model == "next":
                from mmimpress.layer_budget import load_budget_document
                doc = load_budget_document(self.vector_source)
                assert doc["layer_budget_sha256"] == B.VECTOR_RATIO_SHA and doc["alloc_id"] == B.VECTOR
                self._docs = {B.VECTOR: {"path": str(Path(self.vector_source).relative_to(ROOT)), "layer_budget_sha256": doc["layer_budget_sha256"],
                                         "layer_keep_ratios": doc["layer_keep_ratios"], "r_min": doc["r_min"], "r_max": doc["r_max"],
                                         "alloc_id": B.VECTOR, "policy": doc["policy"], "schema": doc["schema"],
                                         "num_layers": doc["num_layers"], "total_ratio": doc["total_ratio"], "source": doc["source"]}}
            else:
                from mmimpress.llava15.store import load_budget
                doc = load_budget(self.vector_source)
                assert doc["layer_budget_sha256"] == B.VECTOR_RATIO_SHA and doc["alloc_id"] == B.VECTOR
                self._docs = {B.VECTOR: dict(doc, path=str(Path(self.vector_source).relative_to(ROOT)))}
        return self._docs

    # ---- model
    def load(self):
        """The loading sequence of the reference harness; the server's cap is 1024 instead of 16."""
        B = self.B
        torch.set_num_threads(2); torch.manual_seed(1234); np.random.seed(1234)
        torch.set_grad_enabled(False)
        from mmimpress.serve import Server
        if self.model == "next":
            runner = B.V.load_runner()
        else:
            from mmimpress.llava15.runner import load_runner
            runner = load_runner(1234)
        server = Server(runner, ratio=B.SERVER_ARGS["ratio"], probe=B.SERVER_ARGS["probe_heads"], alpha=B.SERVER_ARGS["alpha"], max_new_tokens=16)
        warm = B.QA._warmup(runner, server) if self.model == "next" else B.S.warmup(runner, server)      # unmeasured, as in the reference
        server.max_new_tokens = C.MAX_NEW
        server._prefetch_pool()      # the single read-ahead worker thread exists before any request
        self.runner, self.servers = runner, {"base": server}
        self.docs()
        import transformers.models.llama.modeling_llama as ml
        self._clean = {"eager": ml.eager_attention_forward, "pread": os.pread, "preadv": os.preadv, "fadvise": os.posix_fadvise,
                       "write_bytes": Path.write_bytes, "hooks": [len(l._forward_pre_hooks) for l in runner.layers],
                       "forward": [l.self_attn.forward for l in runner.layers]}
        return warm

    def tokenizer_source(self):
        return self.runner.model_id if self.model == "next" else self.runner.source

    def dirty(self):
        """What a failed request left behind (empty list = the process can serve the next request)."""
        import transformers.models.llama.modeling_llama as ml
        from mmimpress import serve
        serve.BIAS.clear()
        c = self._clean
        out = []
        if ml.eager_attention_forward is not c["eager"]:
            out.append("eager attention wrapper not restored")
        if os.pread is not c["pread"] or os.preadv is not c["preadv"] or os.posix_fadvise is not c["fadvise"]:
            out.append("os read / fadvise wrapper not restored")
        if Path.write_bytes is not c["write_bytes"]:
            out.append("Path.write_bytes wrapper not restored")
        if [len(l._forward_pre_hooks) for l in self.runner.layers] != c["hooks"]:
            out.append("layer hook left behind")
        if any(l.self_attn.forward != f for l, f in zip(self.runner.layers, c["forward"])):
            out.append("attention forward not restored")
        return out

    # ---- requests (the calls of the reference harness)
    def pixel(self, image_path, factory, capture):
        B = self.B
        server = self.servers["base"]
        if self.model == "next":
            if capture == B.NEW_KIND:      # the same pixel request, with the attention capture attached
                return B.turn1_request_with_stats(self.runner, server, image_path, factory)
            return B.normal_request(self.runner, server, image_path, factory,
                                    "raster" if capture == "raster" else ("image_only" if capture != "none" else "none"))
        return B.normal_request(self.runner, server, image_path, factory,
                                "raster" if capture in B.FULL_KINDS else ("packed" if capture != "none" else "none"),
                                with_stats=capture == B.NEW_KIND)

    def stored(self, ctx, factory, role, image_id, store_path):
        return self.B.stored_request(self.runner, self.servers, ctx, factory, self.arms[role], image_id, store_path)

    def persist(self, result, diag, path, image_id, kind, source, stats_path):
        if self.model == "next":
            return self.B.persist(self.runner, result, diag, path, image_id, kind, source, self.docs(), stats_path)
        return self.B.persist(self.runner, self.servers["base"], result, diag, path, image_id, kind, source, self.docs(), stats_path)

    def activate(self, kind, path):
        if self.model == "next":
            return self.B.activate(self.runner, kind, path)
        return self.B.activate(self.runner, kind, path, self.docs())

    def forget(self, paths):
        if self.model == "next":      # per-conversation state of the imported harness
            for p in paths:
                self.B.E.TURN1_ORDER_SCORES.pop(str(Path(p)), None)

    def close(self):
        if self.servers:
            for server in self.servers.values():
                server.close()


# ================================================================== requests
def render(G, model, questions, answers):
    after_image, after_eos = G.OFFICIAL_RENDER[model]
    return G.render_prompt(questions, answers, after_image, after_eos)


def token_check(G, tokenizer, slow, image_token_id, prompt, canonical):
    """HF token ids of the rendered prompt == original-LLaVA token ids of the canonical prompt (frozen rule)."""
    ids = tokenizer(prompt).input_ids
    if ids.count(image_token_id) != 1:
        return False, len(ids)
    return G.collapse_image(ids, image_token_id) == G.official_token_ids(slow, canonical), len(ids)


def impress_compact(result, vn):
    log = result["layer_log"]
    only = result.get("importance_only_chunks")
    return {"fb": [int(x["mode"] == "fallback") for x in log], "sim": [x["sim"] for x in log], "kc": [x["k_chunks"] for x in log],
            "vc": [x["v_chunks"] for x in log], "nc": (int(vn) + 63) // 64, "raster": result["raster_chunks"]["chunks"],
            "only": only["chunks"] if only else None, "share": [d["prefix_share"] for d in result["layer_diagnostics"]]}


def make_row(be, phase, role, turn, order, prompt, result, input_tokens, hit, geometry):
    toks = [int(t) for t in result["generated_token_ids"]]
    io = result.get("io", {}) if hit else {}
    per_kind = io.get("per_kind", {})
    wait = result.get("read_wait") or {}
    blocked = wait.get("read_blocked_ms")
    family = be.family(role)
    if hit and blocked is None and family == "ours":
        blocked = result.get("packed_read_ms")      # synchronous packed read: the request thread is blocked for the whole read
    pixel = not hit
    if int(result["vision_forward_count"]) != int(pixel):
        raise AssertionError("vision tower call count does not match the request type")
    if pixel and int(io.get("bytes", 0)) != 0:
        raise AssertionError("a pixel request read from the store")
    drop = DROP_ALWAYS if phase == "gate" else DROP_ALWAYS + DROP_LARGE
    slim = {k: v for k, v in result.items() if k not in drop}
    eos = be.runner.processor.tokenizer.eos_token_id
    row = {"role": role, "method": C.DISPLAY[role], "backend_arm": be.arms[role], "family": family, "turn_id": turn,
           "order_position": order.index(role), "status": "ok",
           "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
           "prediction": result["answer"], "generated_token_ids": toks, "generated_tokens": len(toks),
           "first_token_id": int(result["first_token_id"]), "ended_with_eos": bool(toks and toks[-1] == eos),
           "cap_reached": len(toks) >= C.MAX_NEW, "max_new_tokens": C.MAX_NEW,
           "input_tokens": int(input_tokens), "input_above_4096": int(input_tokens) > C.CONTEXT_LIMIT,
           "official_token_ids_match": True, "request_path": "pixel" if pixel else "stored_kv",
           "ttft_ms": float(result["end_to_end_ttft_ms"]), "request_e2e_ms": float(result["request_e2e_ms"]),
           "prefill_ms": result.get("prefill_ms"), "hook_ms": result.get("hook_ms"), "scatter_ms": result.get("scatter_ms"),
           "ssd_read_bytes": int(io.get("bytes", 0)), "read_calls": int(result.get("pread_calls", 0)) + int(result.get("preadv_calls", 0)),
           "pread_calls": int(result.get("pread_calls", 0)), "preadv_calls": int(result.get("preadv_calls", 0)),
           "probe_read_bytes": int(per_kind.get("probe", {}).get("bytes", 0)), "k_read_bytes": int(per_kind.get("k", {}).get("bytes", 0)),
           "v_read_bytes": int(per_kind.get("v", {}).get("bytes", 0)), "packed_read_bytes": int(per_kind.get("packed", {}).get("bytes", 0)),
           "read_blocked_ms": blocked if hit else None, "completed_at_unix": time.time()}
    if hit and family == "impress_official":
        row["impress"] = impress_compact(result, geometry["v_token_num"])
        row["impress"]["probe_hook_blocked_ms"] = result.get("probe_hook_blocked_ms")
        row["impress"]["prefetch_ready_on_arrival"] = result.get("prefetch_ready_on_arrival")
    ahead = (wait.get("readahead") or {}) if hit else {}
    if ahead:
        row["worker_read_ms"] = ahead.get("worker_read_ms")
    row["result"] = be.B.safe(slim)
    return row


def failed_row(be, role, turn, order, cause, executed):
    return {"role": role, "method": C.DISPLAY[role], "backend_arm": be.arms[role], "family": be.family(role), "turn_id": turn,
            "order_position": order.index(role), "status": "failed", "failure": cause, "executed": bool(executed),
            "prediction": None, "generated_token_ids": None, "generated_tokens": None, "completed_at_unix": time.time()}


def slim_store(p, activation_ms=None):
    keep = ("file_logical_bytes", "allocated_bytes", "bytes", "timing_ms", "native_writer_wall_ms", "service_persistence_ms",
            "completion_full_hash_ms", "completion_publication_ms", "os_write_bytes", "os_write_calls", "write_by_file_name")
    out = {k: p.get(k) for k in keep}
    o = p.get("official_turn1")
    if o:
        out["official_turn1"] = {k: o.get(k) for k in ("capture_ms", "capture_layers", "capture_rows", "materialize_ms", "stats_save_ms",
                                                      "layout_ms", "total_after_request_ms", "suffix_tokens")}
        out["official_turn1"]["layout"] = o.get("layout")
    out["source_role"] = p.get("source", {}).get("role")
    out["activation_ms"] = activation_ms
    return out


def remove_tree(res, path):
    """Delete a directory of this run's scratch area (and nothing else)."""
    path = Path(path)
    if not path.exists():
        return 0
    base = (Path(res) / "scratch").resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(base) or path.resolve() == base:
        raise ValueError("refusing to remove a path outside this run's scratch")
    n = sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())
    shutil.rmtree(path)
    return n


def run_conversation(be, G, slow, conv, phase, res, attempt, tolerate, journal):
    """One conversation, three turns, every arm of the phase.  Returns the commit document (not yet written)."""
    B = be.B
    roles = C.PHASE_ROLES[phase]
    order = C.rotate(roles, conv["ordinal"])
    cid = conv["conversation_id"]
    image_path = ROOT / conv["image_path"]
    image_id = str(conv["image_id"])
    questions = [t["question"] for t in conv["turns"]]
    scratch = Path(res) / "scratch" / be.model / phase / C.safe_name(cid)
    remove_tree(res, scratch)
    scratch.mkdir(parents=True)
    kinds = {r: be.kind(r) for r in roles if be.kind(r) != "none"}
    paths = {k: scratch / k for k in dict.fromkeys(kinds.values())}
    tokenizer = be.runner.processor.tokenizer
    image_token_id = int(be.runner.image_token_id)
    answers = {r: [] for r in roles}
    dead, rows, failures, stores, contexts = {}, [], [], {}, {}
    capturing, built = set(), set()
    geometry = None
    started = time.time()

    def fail(role, turn, stage, exc, executed):
        info = {"conversation_id": cid, "role": role, "method": C.DISPLAY[role], "turn_id": turn, "stage": stage, "attempt": attempt,
                "error_type": type(exc).__name__, "error": str(exc)[:2000], "traceback": traceback.format_exc()[-3000:],
                "at_utc": C.utc(), "oom": isinstance(exc, torch.cuda.OutOfMemoryError)}
        failures.append(info)
        C.append_jsonl(journal, {"event": "failure", **{k: v for k, v in info.items() if k != "traceback"}})
        gc.collect(); torch.cuda.empty_cache()
        left = be.dirty()
        if left:
            raise FatalState(f"{cid} {role} T{turn}: {left}; cause {type(exc).__name__}: {str(exc)[:300]}")
        if not tolerate:
            raise ConversationRetry(json.dumps({k: v for k, v in info.items() if k != "traceback"}, ensure_ascii=False))
        return info

    try:
        B.assert_gpu_exclusive()
        for turn in (1, 2, 3):
            for role in order:
                if role in dead:
                    rows.append(failed_row(be, role, turn, order, f"earlier failure of this method: {dead[role]}", executed=False))
                    continue
                prompt = render(G, be.model, questions[:turn], answers[role])
                canonical = G.canonical_prompt(questions[:turn], answers[role])
                ok, _ = token_check(G, tokenizer, slow, image_token_id, prompt, canonical)      # outside every timer
                if not ok:
                    try:
                        raise ValueError("HF token ids differ from the original-LLaVA token ids (or the history contains an image marker)")
                    except ValueError as exc:
                        fail(role, turn, "official_token_check", exc, executed=False)
                    dead[role] = f"T{turn} official token check"
                    rows.append(failed_row(be, role, turn, order, "official token check failed", executed=False))
                    continue
                factory = lambda p=prompt: (p, "", [])      # noqa: E731  (prompt, history text, history entries of the reference harness)
                kind = kinds.get(role, "none")
                # the first arm of a store kind in the rotation builds that kind's store from its own turn 1
                capture = kind if turn == 1 and kind != "none" and kind not in capturing else "none"
                if capture != "none":
                    capturing.add(capture)
                hit = turn > 1 and role != C.RECOMP
                result = diag = None
                try:
                    if not hit:
                        result, diag = be.pixel(image_path, factory, capture)
                        input_tokens = int(diag["enc_cpu"]["input_ids"].shape[1])
                    else:
                        if kind not in built:
                            raise RuntimeError(f"the store of kind {kind} was not built in turn 1")
                        if kind not in contexts:
                            contexts[kind] = be.activate(kind, paths[kind])
                            stores[kind]["activation_ms"] = contexts[kind].activation["outer_wall_ms"]
                        result, diag = be.stored(contexts[kind], factory, role, image_id, paths[kind])
                        input_tokens = int(contexts[kind].meta["prefix_len"]) + len(diag["suffix_input_ids"])
                    if diag["prompt"] != prompt:
                        raise AssertionError("the request did not use the checked prompt")
                except Exception as exc:      # noqa: BLE001  (recorded; see module docstring)
                    if "GPU_TIMING_CONTAMINATION" in repr(exc):
                        raise
                    result = diag = None
                    fail(role, turn, "request", exc, executed=True)
                    dead[role] = f"T{turn} {type(exc).__name__}"
                    rows.append(failed_row(be, role, turn, order, f"{type(exc).__name__}: {str(exc)[:300]}", executed=True))
                    continue
                if capture != "none":
                    try:
                        source = {"dialog_id": cid, "method_id": be.arms[role], "physical_execution_id": str(uuid.uuid4()),
                                  "logical_request_id": f"{C.SCHEMA}:{Path(res).name}:{be.model}:{phase}:{role}:{cid}:t1", "role": role}
                        p = be.persist(result, diag, paths[capture], image_id, capture, source, scratch / f"{capture}_turn1_stats.npz")
                        stores[capture] = slim_store(p)
                        meta = json.loads((paths[capture] / "meta.json").read_text())
                        g = {k: meta[k] for k in ("num_heads", "head_dim", "num_layers", "v_token_start", "v_token_num", "n_spatial", "prefix_len")}
                        g["newline"] = len(meta.get("newline_idx", []))
                        if geometry is None:
                            geometry = g
                        elif geometry != g:
                            raise AssertionError("the stores of one conversation disagree on the prefix geometry")
                        built.add(capture)
                        del meta, p
                    except Exception as exc:      # noqa: BLE001
                        fail(role, turn, f"store_build:{capture}", exc, executed=True)
                result.pop("captured_past_key_values", None)
                row = make_row(be, phase, role, turn, order, prompt, result, input_tokens, hit, geometry)
                rows.append(row)
                answers[role].append(row["prediction"])
                C.append_jsonl(journal, {"event": "row", "role": role, "turn_id": turn, "generated_tokens": row["generated_tokens"],
                                         "ttft_ms": row["ttft_ms"], "at_unix": row["completed_at_unix"]})
                del result, diag
        B.assert_gpu_exclusive()
    finally:
        for ctx in contexts.values():      # file descriptors closed before the stores are removed
            try:
                ctx.close()
            except Exception:      # noqa: BLE001
                pass
        contexts.clear()
        be.forget(paths.values())
        gc.collect()
    store_bytes = {k: sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) for k, path in paths.items() if path.exists()}
    main_t1 = [tuple(r["generated_token_ids"]) for r in rows if r["turn_id"] == 1 and r["role"] in C.MAIN_ROLES and r["status"] == "ok"]
    return {"schema": C.SCHEMA, "model": be.model, "phase": phase, "conversation_id": cid, "source_id": conv["source_id"],
            "source_row_index": conv["source_row_index"], "ordinal": conv["ordinal"], "image_id": image_id,
            "image_sha256": C.sha(image_path), "roles": roles, "order": order, "attempt": attempt, "rows": rows,
            "failures": failures, "stores": stores, "store_bytes_on_disk": store_bytes, "geometry": geometry,
            "t1_tokens_identical_main_methods": len(main_t1) == len(C.MAIN_ROLES) and len(set(main_t1)) == 1,
            "seconds": time.time() - started, "committed_at_utc": C.utc(), "pid": os.getpid(), "boot_id": C.boot_id()}


# ================================================================== phase loop
def wait_for_gpu(be, res, phase):
    waited = 0
    while True:
        try:
            be.B.assert_gpu_exclusive()
            return waited
        except RuntimeError as exc:
            if waited >= GPU_WAIT_SECONDS:
                raise
            if waited == 0:
                STATUS.update(C.res_rel(res), last_error={"stage": f"{be.model}_{phase}", "error": repr(exc)[:300], "at_utc": C.utc(),
                                                          "note": "waiting for the other process to leave the GPU"})
            time.sleep(30); waited += 30


def capacity_guard(res, config):
    free = shutil.disk_usage(res).free
    need = int(config["storage"]["required_free_bytes"])
    if free < need:
        raise RuntimeError(f"BLOCKED_STORAGE: free={free}; required={need}")
    return free


def discard_partial(res, model, phase, cid, attempt, mode, extra=None):
    """Move the journal of an abandoned attempt aside and remove its stores; write the resume-log line."""
    pdir = C.phase_dir(res, model, phase)
    journal = pdir / "partial" / f"{C.safe_name(cid)}.attempt{attempt}.jsonl"
    rows = sum(1 for e in C.read_jsonl(journal) if e.get("event") == "row")
    if journal.exists():
        (pdir / "discarded").mkdir(exist_ok=True)
        os.replace(journal, pdir / "discarded" / f"{journal.name}.{int(time.time())}")
    removed = remove_tree(res, Path(res) / "scratch" / model / phase / C.safe_name(cid))
    event = {"at_utc": C.utc(), "at_unix": time.time(), "model": model, "phase": phase, "conversation_id": cid, "mode": mode,
             "attempt_abandoned": attempt, "rows_discarded": rows, "store_bytes_removed": removed, "pid": os.getpid(), "boot_id": C.boot_id()}
    event.update(extra or {})
    C.append_jsonl(Path(res) / "resume_log.jsonl", event)
    return event


def run_phase(be, res, phase, config):
    model = be.model
    G = gen_module()
    from transformers import AutoTokenizer
    slow = AutoTokenizer.from_pretrained(be.tokenizer_source(), use_fast=False, legacy=True)
    index, _ = C.load_index()
    convs = C.phase_conversations(res, model, phase, index)
    if convs is None:
        raise SystemExit("the overflow selection does not exist yet")
    pdir = C.phase_dir(res, model, phase)
    (pdir / "conversations").mkdir(parents=True, exist_ok=True)
    (pdir / "partial").mkdir(exist_ok=True)
    marker = pdir / "in_progress.json"
    attempts_path = pdir / "attempts.json"
    attempts = C.read_json(attempts_path, {})
    rel = C.res_rel(res)

    # a conversation the previous process left unfinished: nothing of it is reused
    left = C.read_json(marker)
    if left is not None:
        if not C.conv_path(res, model, phase, left["conversation_id"]).exists():
            discard_partial(res, model, phase, left["conversation_id"], left["attempt"], "interrupted_conversation_restarts_at_turn1",
                            {"interrupted_process": left.get("pid"), "interrupted_boot_id": left.get("boot_id"), "started_utc": left.get("started_utc")})
            attempts[left["conversation_id"]] = max(0, int(attempts.get(left["conversation_id"], 1)) - 1)      # the interrupted attempt does not count
            C.atomic_json(attempts_path, attempts)
        marker.unlink()
    scratch_root = Path(res) / "scratch" / model / phase
    if scratch_root.is_dir():      # stores of a conversation that was committed but not cleaned up
        for folder in sorted(scratch_root.iterdir()):
            remove_tree(res, folder)

    done = {c["conversation_id"] for c in convs if C.conv_path(res, model, phase, c["conversation_id"]).exists()}
    failed_convs = 0
    for c in convs:
        if c["conversation_id"] in done:
            failed_convs += bool(C.read_gz_json(C.conv_path(res, model, phase, c["conversation_id"]))["failures"])
    consecutive = 0
    started = time.time(); new = 0

    def status(current=None):
        now = time.time()
        rate = (now - started) / new if new else None
        left_n = len(convs) - len(done)
        ssd = STATUS.ssd_counter(rel, f"{model}_{phase}")
        temp = STATUS.ssd_temperature()
        st = STATUS.read(rel)
        after = float(st.get("estimated_seconds_after_this_stage") or 0.0)
        fields = dict(model=C.MODEL_NAME[model], phase=phase, stage=f"{model}_{phase}", conversations_done=len(done), conversations_total=len(convs),
                      conversations_with_failed_requests=failed_convs, current_conversation=current,
                      seconds_per_conversation_this_process=rate, stage_eta_utc=C.utc(now + rate * left_n) if rate else None,
                      eta_utc=C.utc(now + rate * left_n + after) if rate else st.get("eta_utc"),
                      ssd_written_bytes=ssd.get("written_bytes"), ssd_written_gb=round((ssd.get("written_bytes") or 0) / 1e9, 2),
                      ssd_written_note="device-wide kernel counter since the driver first started (all processes); not NAND writes",
                      ssd_temperature_c=temp["composite_c"], free_gib=round(shutil.disk_usage(res).free / GiB, 1), python_pid=os.getpid())
        STATUS.update(rel, **fields)

    status()
    STATUS.ssd_sample(rel, f"{model}_{phase}_process_start", model=model, phase=phase, committed=len(done))
    for conv in convs:
        cid = conv["conversation_id"]
        if cid in done:
            continue
        while True:
            capacity_guard(res, config)
            wait_for_gpu(be, res, phase)
            attempt = int(attempts.get(cid, 0)) + 1
            attempts[cid] = attempt
            C.atomic_json(attempts_path, attempts)
            journal = pdir / "partial" / f"{C.safe_name(cid)}.attempt{attempt}.jsonl"
            C.atomic_json(marker, {"conversation_id": cid, "attempt": attempt, "pid": os.getpid(), "boot_id": C.boot_id(), "started_utc": C.utc()})
            t0 = time.time()
            try:
                doc = run_conversation(be, G, slow, conv, phase, res, attempt, tolerate=attempt >= 2, journal=journal)
            except ConversationRetry as exc:
                discard_partial(res, model, phase, cid, attempt, "request_failed_conversation_restarts_at_turn1", {"failure": str(exc)[:1500]})
                marker.unlink()
                STATUS.update(rel, last_error={"stage": f"{model}_{phase}", "conversation_id": cid, "error": str(exc)[:500], "at_utc": C.utc(),
                                               "note": "request failed; the conversation is run once more from turn 1"})
                continue
            except RuntimeError as exc:
                if "GPU_TIMING_CONTAMINATION" not in repr(exc):
                    raise
                discard_partial(res, model, phase, cid, attempt, "gpu_used_by_another_process_conversation_restarts_at_turn1", {"error": repr(exc)[:500]})
                marker.unlink()
                attempts[cid] = attempt - 1
                C.atomic_json(attempts_path, attempts)
                torch.cuda.empty_cache()
                continue
            doc["config_sha256"] = config["_file_sha256"]
            C.atomic_gz_json(C.conv_path(res, model, phase, cid), doc)
            remove_tree(res, Path(res) / "scratch" / model / phase / C.safe_name(cid))      # the conversation's stores
            journal.unlink(missing_ok=True)
            marker.unlink()
            torch.cuda.empty_cache()
            break
        done.add(cid); new += 1
        if doc["failures"]:
            failed_convs += 1; consecutive += 1
            STATUS.update(rel, last_error={"stage": f"{model}_{phase}", "conversation_id": cid, "at_utc": C.utc(),
                                           "error": "; ".join(f"{f['method']} T{f['turn_id']} {f['stage']}: {f['error_type']}" for f in doc["failures"])[:500],
                                           "note": "committed with the failed request(s) recorded"})
        else:
            consecutive = 0
        status(cid)
        STATUS.ssd_sample(rel, "conversation_committed", model=model, phase=phase, conversation_id=cid, committed=len(done),
                          seconds=time.time() - t0, store_bytes=sum(doc["store_bytes_on_disk"].values()),
                          store_write_bytes=sum(int(s.get("os_write_bytes") or 0) for s in doc["stores"].values()))
        print(json.dumps({"model": model, "phase": phase, "done": len(done), "of": len(convs), "conversation": cid,
                          "seconds": round(time.time() - t0, 1), "generated": [r["generated_tokens"] for r in doc["rows"]],
                          "failures": len(doc["failures"]), "t1_identical": doc["t1_tokens_identical_main_methods"]}), flush=True)
        if consecutive >= C.MAX_CONSECUTIVE_FAILED_CONVERSATIONS or failed_convs > C.MAX_FAILED_CONVERSATIONS:
            raise RuntimeError(f"TOO_MANY_FAILED_CONVERSATIONS: {failed_convs} conversations with failed requests, {consecutive} in a row")
    assert len(done) == len(convs)
    C.atomic_json(pdir / "DONE.json", {"status": "COMPLETE", "model": model, "phase": phase, "conversations": len(convs),
                                       "requests": len(convs) * 3 * len(C.PHASE_ROLES[phase]),
                                       "conversations_with_failed_requests": failed_convs, "finished_utc": C.utc(), "config_sha256": config["_file_sha256"]})


# ================================================================== freeze / prerun
def reference_checks(be):
    """The imported harness is the code and the settings of the reference runs (asserted, and recorded)."""
    B = be.B
    now = C.source_hashes()
    out = {}
    if be.model == "next":
        impl = json.loads((B.IMPL_REF_RUN / "config.json").read_text()); ours = json.loads((B.OURS_REF_RUN / "config.json").read_text())
        latest = json.loads((ROOT / "runs/llava_next_mtvqa_main_20261121T063602Z/config.json").read_text())
        assert impl["server_args"] == B.SERVER_ARGS == ours["server_args"] and impl["impress_official"] == B.OFFICIAL
        assert impl["readahead"][B.FULL_ASYNC] == B.READAHEAD[B.FULL_ASYNC] and impl["readahead"][B.NEW_ASYNC] == B.READAHEAD[B.NEW_ASYNC]
        assert ours["readahead"][B.ORIG_ASYNC] == B.READAHEAD[B.ORIG_ASYNC] and impl["readahead"]["max_ahead"] == ours["readahead"]["max_ahead"] == B.READAHEAD["max_ahead"]
        assert impl["model"] == ours["model"] == B.MODEL_ID and impl["revision"] == ours["revision"] == B.REVISION
        strip = lambda a: {k: v for k, v in a.items() if k != "display"}      # noqa: E731
        assert strip(ours["arms"][B.ORIG_ASYNC]) == strip(B.ARMS[B.ORIG_ASYNC])
        for m in (B.FULL_ASYNC, B.NEW, B.NEW_ASYNC, B.NEW_R100, B.FULL_REORDERED):
            assert strip(impl["arms"][m]) == strip(B.ARMS[m]), m
        for name, ref, allowed in (("mt_gqa_impress_code_20261117", impl, B.CHANGED_SINCE_IMPL_REF), ("mt_gqa_ours_order_20261120", ours, B.CHANGED_SINCE_OURS_REF),
                                   ("llava_next_mtvqa_main_20261121", latest, ())):
            before = {k: v for k, v in ref["source_sha256"].items() if not C.REPORT_SCRIPT.match(k)}
            changed = sorted(k for k in before if now.get(k) != before[k])
            assert set(changed) <= set(allowed), (name, changed)
            out[name] = {"files_compared": len(before), "changed": changed}
        env = impl["environment"]
    else:
        ref = json.loads((ROOT / "runs/llava15_mt_gqa_main_20261120T165001Z/config.json").read_text())
        assert ref["server_args"] == B.SERVER_ARGS and ref["impress_official"] == B.OFFICIAL and ref["readahead"] == B.READAHEAD
        assert ref["model"] == B.MODEL_ID and ref["revision"] == B.REVISION
        for m in be.arms.values():
            assert ref["arms"][m] == dict(B.ARMS[m]), m
        before = {k: v for k, v in ref["source_sha256"].items() if not C.REPORT_SCRIPT.match(k)}
        changed = sorted(k for k in before if now.get(k) != before[k])
        assert not changed, changed
        out["llava15_mt_gqa_main_20261120"] = {"files_compared": len(before), "changed": changed}
        env = ref["environment"]
    import transformers
    assert (transformers.__version__, torch.__version__, np.__version__, sys.executable) == (
        env["transformers"], env["torch"], env["numpy"], env["executable"]), "not the environment of the reference runs"
    return out


def freeze(res):
    import transformers
    res = Path(res)
    assert res.parent == ROOT / "results" and res.name.startswith(C.RUN_PREFIX) and C.CONTRACT.is_file()
    if (res / "config.json").exists():
        raise FileExistsError("config.json exists: the run is already frozen")
    G = gen_module()
    index, index_sha = C.load_index()
    models = {}
    for model in C.MODELS:
        be = Backend(model)
        models[model] = dict(be.describe(), reference_checks=reference_checks(be), render_after_image=G.OFFICIAL_RENDER[model][0],
                             render_after_eos=G.OFFICIAL_RENDER[model][1],
                             alignment_generation=str(C.ALIGN_GEN[model].relative_to(ROOT)), alignment_judge=str(C.ALIGN_JUDGE[model].relative_to(ROOT)))
        assert (C.ALIGN_GEN[model] / "DONE").exists() and (C.ALIGN_JUDGE[model] / "DONE").exists()
    assert G.NOMINAL_CAP == C.MAX_NEW
    # the largest image bundle: LLaVA-NeXT AnyRes at its largest grid; refined per image by the prerun stage
    bundle = int(Backend("next").B.store_bytes(2880, 48)["bundle"])
    config = {
        "schema": C.SCHEMA, "frozen_utc": C.utc(), "contract": str(C.CONTRACT.relative_to(ROOT)), "contract_sha256": C.sha(C.CONTRACT),
        "alignment_doc_sha256": C.sha(C.ALIGN_DOC), "alignment_run": str(C.ALIGN.relative_to(ROOT)), "index_sha256": index_sha,
        "models": models, "model_order": list(C.MODELS),
        "generation": {"max_new_tokens": C.MAX_NEW, "cap_policy": "always 1024, no context adjustment", "decoding": "greedy (do_sample=False)",
                       "precision": "NF4 4-bit (double quant), BF16 compute, eager attention", "kv_store_dtype": "float16",
                       "prompt": "llava_v1 template rendered so that the HF token ids equal the original-LLaVA token ids; checked before every request",
                       "history": "the same method's generated answers; image in the first turn only",
                       "context": "prompt and history are never truncated; inputs above 4096 tokens run as they are",
                       "renderer": "scripts/213_convbench_recomp_generate.py (render_prompt, OFFICIAL_RENDER, official_token_ids)",
                       "renderer_sha256": C.sha(ROOT / "scripts/213_convbench_recomp_generate.py")},
        "judge": {"cap": C.JUDGE_CAP, "script": "scripts/219_convbench_main_judge.py (the functions of scripts/214)",
                  "view": "capN (first-pass cap 2048 + official extraction + official Rating fallback)"},
        "statistics": {"unit": "conversation", "resamples": C.RESAMPLES, "seed": C.SEED, "ci": 0.95},
        "roles": {"main": C.MAIN_ROLES, "gate": C.GATE_ROLES, "display": C.DISPLAY},
        "phases": {"gate": {"ordinals": list(C.GATE_ORDINALS), "roles": C.GATE_ROLES},
                   "pilot": {"ordinals": list(C.PILOT_ORDINALS), "roles": C.MAIN_ROLES},
                   "overflow": {"roles": C.MAIN_ROLES, "selection": "after the pilot (gates stage): only when the pilot had no input above 4096 tokens"},
                   "main": {"ordinals": "0..576 (index order)", "roles": C.MAIN_ROLES}},
        "conversation_ids": {"gate": [index[o]["conversation_id"] for o in C.GATE_ORDINALS],
                             "pilot": [index[o]["conversation_id"] for o in C.PILOT_ORDINALS]},
        "arm_rotation": "arm order of a conversation = the phase's arm list rotated by (index ordinal % number of arms)",
        "expected_requests": {"main_per_model": C.N_CONV * 3 * 4, "main_total": 2 * C.N_CONV * 3 * 4},
        "failure_policy": {"first_failure": "the conversation starts again at turn 1 for every arm",
                           "second_failure": "committed with the failed request recorded; later turns of that arm are failed; denominators stay 577",
                           "stop": {"consecutive_conversations": C.MAX_CONSECUTIVE_FAILED_CONVERSATIONS, "total_conversations": C.MAX_FAILED_CONVERSATIONS}},
        "storage": {"largest_bundle_bytes_bound": bundle, "free_space_factor": FREE_SPACE_FACTOR, "required_free_bytes": FREE_SPACE_FACTOR * bundle,
                    "policy": "one live conversation; its stores are deleted when it is committed",
                    "deletion_scope": "results/<run>/scratch only"},
        "source_sha256": C.source_hashes(),
        "environment": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__, "transformers": transformers.__version__,
                        "numpy": np.__version__},
    }
    C.atomic_json(res / "config.json", config)
    shutil.copy2(C.CONTRACT, res / "CONTRACT.md")
    print(json.dumps({"freeze": "PASS", "run": str(res), "source_files": len(config["source_sha256"]),
                      "reference_checks": {m: models[m]["reference_checks"] for m in models}}), flush=True)


def next_geometry(index):
    """Number of visual tokens and of row separators of every image under LLaVA-NeXT AnyRes (processor only, no model)."""
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.image_processing_utils import select_best_resolution
    B = Backend("next").B
    proc = AutoProcessor.from_pretrained(B.MODEL_ID)
    ip = proc.image_processor
    size = ip.crop_size["height"] if isinstance(ip.crop_size, dict) else int(ip.crop_size)
    patches = size // proc.patch_size
    out = {}
    for conv in index:
        if conv["image_path"] in out:
            continue
        with Image.open(ROOT / conv["image_path"]) as im:
            w, h = im.size
        best_h, best_w = select_best_resolution([h, w], ip.image_grid_pinpoints)
        unpadded, newline = proc._get_unpadded_features(h, w, patches, patches, best_h // size, best_w // size)
        base = patches * patches      # the base (low-resolution) view
        out[conv["image_path"]] = {"n_content": int(unpadded + base), "newline": int(newline), "width": w, "height": h}
    return out


def prerun(res):
    """Checks before any GPU measurement; writes <res>/prerun.json."""
    res = Path(res)
    config = C.verify_freeze(res)
    index, _ = C.load_index()
    started = time.time()
    geo = next_geometry(index)
    nb = Backend("next").B
    lb = Backend("llava15").B
    # LLaVA-1.5: fixed geometry (576 tokens); the store sizes the reference run recorded
    l15_full = (2 * lb.NUM_LAYERS * lb.N_VISUAL * lb.S.ROW_BYTES + lb.NUM_LAYERS * lb.N_VISUAL * lb.SERVER_ARGS["probe_heads"] * 128 * 2
                + lb.NUM_LAYERS * lb.OFFICIAL["probe_heads"] * lb.N_VISUAL * 4 + (2 << 20))
    l15_packed = lb.PACKED_BYTES + (2 << 20) + 2_622_802
    l15 = {"fullload_store": l15_full, "impress_store": l15_full, "ours_store": l15_packed, "written": 3 * l15_full + l15_packed,
           "bundle": 2 * l15_full + l15_packed}
    per_conv = {}
    for conv in index:
        g = geo[conv["image_path"]]
        b = nb.store_bytes(g["n_content"], g["newline"])
        x = CONVBENCH_SYSTEM_PREFIX_EXTRA_BYTES
        per_conv[conv["conversation_id"]] = {"next": {"fullload_store": b["raster"] + x, "impress_store": b[nb.NEW_KIND] + x, "ours_store": b[nb.ORIG_KIND] + x,
                                                      "written": b["written"] + 3 * x, "bundle": b["bundle"] + 3 * x},
                                             "llava15": {k: v + (3 * x if k in ("written", "bundle") else x) for k, v in l15.items()}}
    phase_ordinals = {"gate": list(C.GATE_ORDINALS), "pilot": list(C.PILOT_ORDINALS), "main": list(range(C.N_CONV))}
    writes = {}
    for model in C.MODELS:
        writes[model] = {ph: int(sum(per_conv[index[o]["conversation_id"]][model]["written"] for o in ords)) for ph, ords in phase_ordinals.items()}
        mean_w = float(np.mean([per_conv[c["conversation_id"]][model]["written"] for c in index]))
        writes[model]["overflow_if_run"] = int(C.OVERFLOW_CONVERSATIONS * mean_w)
        writes[model]["total"] = int(sum(writes[model].values()))
    total = sum(writes[m]["total"] for m in C.MODELS)
    largest = max(max(v[m]["bundle"] for m in C.MODELS) for v in per_conv.values())
    free = shutil.disk_usage(res).free
    required = FREE_SPACE_FACTOR * largest
    # time: ReComp-only seconds per conversation of the alignment run x 4 arms + store writing; judge from the alignment run
    seconds = {}
    for model in C.MODELS:
        s4 = ALIGN_SECONDS_PER_CONVERSATION[model] * 4 + (12.0 if model == "next" else 3.0)
        s9 = ALIGN_SECONDS_PER_CONVERSATION[model] * 9 + (12.0 if model == "next" else 3.0)
        seconds[model] = {"per_conversation_4_arms_estimate": s4, "gate": 4 * s9 + 120, "pilot": 40 * s4 + 120,
                          "overflow_if_run": 2 * s4 * 2 + 120, "main": C.N_CONV * s4 + 120}
        seconds[model]["total"] = sum(v for k, v in seconds[model].items() if k != "per_conversation_4_arms_estimate")
    # every stage prompt of the official judge holds all three answers, so a method is judged afresh unless its three answers equal
    # another method's; low: ReComp read from the alignment cache and part of FullLoad equal to ReComp; high: four fresh answer sets
    judge_low = 2 * ALIGN_JUDGE_SECONDS_PER_ANSWER_SET * 2.6
    judge_high = 2 * ALIGN_JUDGE_SECONDS_PER_ANSWER_SET * 4.0
    fixed = 20 * 60 + 15 * 60 + 10 * 60      # tests, judge check, reports
    low = sum(seconds[m]["total"] for m in C.MODELS) + judge_low + fixed
    high = sum(seconds[m]["total"] for m in C.MODELS) * 1.25 + judge_high + fixed
    align = {}
    for model in C.MODELS:
        n_rows = 0; missing = []
        for conv in index:
            try:
                rows = C.alignment_rows(model, conv["conversation_id"])
                n_rows += sum(1 for r in rows if isinstance(r.get("prediction"), str) and r.get("generated_tokens") is not None)
            except Exception as exc:      # noqa: BLE001
                missing.append({"conversation_id": conv["conversation_id"], "error": repr(exc)[:200]})
        over = []
        for conv in index:
            rows = C.alignment_rows(model, conv["conversation_id"])
            mx = max(r["input_tokens"] for r in rows)
            over.append((mx, conv["ordinal"], conv["conversation_id"]))
        align[model] = {"recomp_rows": n_rows, "expected": 3 * C.N_CONV, "missing": missing[:20],
                        "requests_above_4096": sum(1 for conv in index for r in C.alignment_rows(model, conv["conversation_id"]) if r["input_above_4096"]),
                        "conversations_by_largest_input": [{"input_tokens": a, "ordinal": b, "conversation_id": c} for a, b, c in sorted(over, reverse=True)[:8]],
                        "token_ids_recorded": False,
                        "note": "the alignment rows hold the answer text, the number of generated tokens and the first token id (no token id list)"}
    smart = STATUS.smart_snapshot()
    blocking = []
    if free < required:
        blocking.append(f"free space {free} B is below {FREE_SPACE_FACTOR} x the largest per-conversation store bundle ({required} B)")
    for model in C.MODELS:
        if align[model]["recomp_rows"] != 3 * C.N_CONV:
            blocking.append(f"alignment ReComp record of {model} is incomplete")
    doc = {
        "written_before_any_gpu_measurement_utc": C.utc(), "run": res.name, "status": "PASS" if not blocking else "BLOCKED", "blocking": blocking,
        "expected_ssd_write": {
            "definition": "per conversation, the bytes the three store builds write: FullLoad store + IMPRESS store (written, then rewritten in "
                          "the reordered layout) + Ours store. LLaVA-NeXT: from each image's AnyRes geometry; LLaVA-1.5: fixed 576 tokens. "
                          "Request rows, status files and the judge cache are not included (well below 1 % of the total).",
            "bytes": writes, "total_bytes": int(total), "total_tb": total / 1e12,
            "per_model_tb": {m: writes[m]["total"] / 1e12 for m in C.MODELS},
            "per_conversation_mean_bytes": {m: {k: float(np.mean([per_conv[c["conversation_id"]][m][k] for c in index]))
                                                for k in ("fullload_store", "impress_store", "ours_store", "written")} for m in C.MODELS},
            "largest_conversation_bundle_bytes": int(largest)},
        "expected_duration": {
            "seconds": seconds, "judge_seconds_low": judge_low, "judge_seconds_high": judge_high,
            "total_hours_low": low / 3600, "total_hours_high": high / 3600,
            "basis": "generation: ReComp-only seconds per conversation of the alignment run (LLaVA-NeXT 13.2 s, LLaVA-1.5 8.1 s) x number of arms "
                     "+ store writing; judging: 3.6 h per fresh answer set (577 x 3 judgments) in the alignment run, 2.6 - 4 fresh answer sets per "
                     "model. Replaced by the pilot's measured seconds per conversation in status.json when each pilot has finished."},
        "free_space": {"free_bytes": free, "free_gib": free / GiB, "largest_conversation_bundle_bytes": int(largest),
                       "required_free_bytes": int(required), "ratio_free_to_bundle": free / largest,
                       "rule": f"do not start when free space < {FREE_SPACE_FACTOR} x the largest per-conversation store bundle; rechecked before every conversation",
                       "ok": free >= required},
        "smart": smart,
        "ssd_counters_at_prerun": STATUS._safe(STATUS.sectors), "ssd_temperature_at_prerun": STATUS.ssd_temperature(),
        "store_deletion": "the stores of a conversation are deleted as soon as the conversation is committed (one live conversation at a time)",
        "alignment_record": align,
        "llava_next_image_geometry": {"images": len(geo), "visual_tokens_min": min(g["n_content"] + g["newline"] for g in geo.values()),
                                      "visual_tokens_max": max(g["n_content"] + g["newline"] for g in geo.values()),
                                      "visual_tokens_mean": float(np.mean([g["n_content"] + g["newline"] for g in geo.values()]))},
        "seconds": time.time() - started}
    C.atomic_json(res / "prerun.json", doc)
    C.atomic_json(res / "prerun_receipt.json", {"status": doc["status"], "blocking": blocking, "prerun_sha256": C.sha(res / "prerun.json"),
                                                "config_sha256": config["_file_sha256"], "at_utc": C.utc()})
    STATUS.update(C.res_rel(res), expected_ssd_write_tb=round(total / 1e12, 2), expected_total_hours=[round(low / 3600, 1), round(high / 3600, 1)],
                  smart=smart["status"], prerun=doc["status"])
    print(json.dumps({"prerun": doc["status"], "blocking": blocking, "expected_ssd_write_tb": round(total / 1e12, 2),
                      "hours": [round(low / 3600, 1), round(high / 3600, 1)], "free_gib": round(free / GiB, 1), "smart": smart["status"]}), flush=True)
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=Path, required=True, help="results/convbench_main_<UTC>")
    p.add_argument("--stage", choices=("freeze", "prerun") + C.PHASES, required=True)
    p.add_argument("--model", choices=C.MODELS)
    args = p.parse_args()
    res = args.res.resolve()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.stage == "freeze":
        return freeze(res)
    if args.stage == "prerun":
        doc = prerun(res)
        sys.exit(0 if doc["status"] == "PASS" else 1)
    if args.model is None:
        p.error("--model is required for a request phase")
    config = C.verify_freeze(res)
    receipt = C.read_json(res / "prerun_receipt.json") or {}
    if receipt.get("status") != "PASS" or receipt.get("config_sha256") != config["_file_sha256"]:
        raise RuntimeError("PRERUN_NOT_PASSED")
    phase, model = args.stage, args.model
    if phase == "main":      # the main phase is admitted only by the recorded gate decision of this model
        admit = C.read_json(res / "gates" / model / "admission.json") or {}
        if admit.get("status") != "PASS" or admit.get("config_sha256") != config["_file_sha256"]:
            raise RuntimeError("MAIN_NOT_ADMITTED")
    if C.phase_complete(res, model, phase):
        print(json.dumps({"phase": phase, "model": model, "status": "already complete"}))
        return
    be = Backend(model)
    wait_for_gpu(be, res, phase)
    be.load()
    C.append_jsonl(res / "process_log.jsonl", {"event": "phase_process_start", "model": model, "phase": phase, "pid": os.getpid(),
                                               "boot_id": C.boot_id(), "at_utc": C.utc()})
    try:
        run_phase(be, res, phase, config)
    except FatalState as exc:
        STATUS.update(C.res_rel(res), last_error={"stage": f"{model}_{phase}", "error": repr(exc)[:500], "at_utc": C.utc(),
                                                  "note": "process state not clean after a failed request; the phase process is restarted"})
        print(json.dumps({"fatal_state": repr(exc)[:500]}), flush=True)
        os._exit(75)
    except BaseException as exc:
        STATUS.update(C.res_rel(res), last_error={"stage": f"{model}_{phase}", "error": repr(exc)[:500], "at_utc": C.utc()})
        print(json.dumps({"error": repr(exc)[:1000], "traceback": traceback.format_exc()[-3000:]}), flush=True)
        raise
    finally:
        be.close()


if __name__ == "__main__":
    main()
