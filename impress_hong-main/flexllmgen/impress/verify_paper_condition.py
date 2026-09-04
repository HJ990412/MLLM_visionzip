"""Paper §6.1-faithful verification (design: IMPRESS_paper_condition.md).

Every parameter is set INDEPENDENTLY to the paper's stated value; only the
GPU (RTX 4090 24GB vs A100 80GB) differs physically:
  - datasets RTE / COPA (LM-Evaluation-Harness tasks)      [paper: 4 sets]
  - 2..10 few-shot examples per prefix, uniform            [exact]
  - prefix length ~ U(4.6k, 5.9k), mean ~5.25k, cap 10K    [paper: 4.8-5.7k
    avg, 10K cap for non-30B models]. Standard QA templates cannot reach
    this with 2-10 shots; the paper itself says "we extend the prefixes".
    ESTIMATED mechanism: each shot carries a long document-style context
    passage (contiguous, non-repeating wikitext slices), InfiniGen-style.
  - prefix-KV pool: RTE 57GB / COPA 64GB, verbatim         [exact]
  - retention: accuracy sweep 50/25/10/5%; TTFT fixed at
    COPA 50% / RTE 25%                                     [exact, §6.2]
  - chunk size 64 tokens                                   [exact]
  - caches: GPU 10GB attempted first (paper value); on OOM re-run with the
    computed 4090 maximum via --gpu-cache-gb; CPU 32GB     [paper absolute]
  - reuse frequency: normal distribution over the pool     [exact]
  - metric: accuracy only                                  [exact]

Protocol per dataset: build pool -> warm-up pass -> one synchronous KV
reorder (the experimental stand-in for the paper's "every 10 minutes") ->
(a) retention-accuracy sweep -> (b) fixed-retention TTFT, 3 modes
(IMPRESS -> free GPU cache tier -> FullLoad -> ReComp; ReComp doubles as
the accuracy reference).

Usage:
  python -m flexllmgen.impress.verify_paper_condition --dataset rte
  python -m flexllmgen.impress.verify_paper_condition --dataset copa
  # after an OOM at 10GB:
  ... --dataset rte --gpu-cache-gb 8 --keep-tree
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch

from flexllmgen.impress.impress_serving import (ImpressConfig, ImpressServer,
    ChunkKVManager)
from flexllmgen.impress.selective_loading import PathKVProvider

BYTES_PER_TOKEN = 32 * 2 * 4096 * 2   # OPT-6.7B: layers x {K,V} x h x fp16
PREFIX_LEN_RANGE = (4600, 5900)       # mean ~5.25k (paper avg 4.8-5.7k)
PREFIX_CAP = 10 * 1024                # paper: 10K for non-30B OPT models
RETENTION_SWEEP = [0.50, 0.25, 0.10, 0.05]

DATASET_CFG = dict(
    rte=dict(pool_gb=57.0, retention_ttft=0.25, num_questions=120),
    copa=dict(pool_gb=64.0, retention_ttft=0.50, num_questions=100),
)


# ------------------------------------------------------------- workload
def load_passage_pool():
    """Contiguous long-document text for shot contexts (never repeated:
    each slice is consumed once across the whole pool)."""
    from datasets import load_dataset
    d = load_dataset("wikitext", "wikitext-2-raw-v1")
    text = " ".join(t.strip() for t in d["train"]["text"] if t.strip())
    return text.split(" ")


class PassageCursor:
    def __init__(self, words):
        self.words = words
        self.pos = 0

    def take(self, tokenizer, n_tokens):
        """Next unread slice of ~n_tokens tokens (words are ~1.3 tokens)."""
        n_words = max(8, int(n_tokens / 1.15))  # measured wikitext BPE ratio
        chunk = " ".join(self.words[self.pos:self.pos + n_words])
        self.pos += n_words
        assert self.pos < len(self.words), "passage pool exhausted"
        return chunk


def rte_shot(ex, ans=True):
    from flexllmgen.impress.verify_selection_rte import format_rte
    s = format_rte(ex)
    return s + (" True" if ex["label"] == 0 else " False") if ans else s


def copa_context(ex):
    conn = " because" if ex["question"] == "cause" else " so"
    return ex["premise"].strip().rstrip(".") + conn


def copa_cont(choice):
    c = choice.strip()
    return " " + c[0].lower() + c[1:]


def build_workload(dataset, tokenizer, seed):
    """Returns (prefix_token_lists, question_items, assignment)."""
    from datasets import load_dataset
    cfg = DATASET_CFG[dataset]
    rng = np.random.RandomState(seed)
    words = load_passage_pool()
    cursor = PassageCursor(words)

    if dataset == "rte":
        d = load_dataset("glue", "rte")
    else:
        d = load_dataset("super_glue", "copa", trust_remote_code=True)
    train = d["train"]
    val = d["validation"]

    # ---- prefixes: accumulate until the pool hits the paper's GB target
    target_bytes = cfg["pool_gb"] * 1e9
    prefixes, used_bytes = [], 0
    shot_pool = rng.permutation(len(train)).tolist()
    sp = 0
    while used_bytes < target_bytes - 1.4e9:  # stop within ~half a prefix
        target_len = int(rng.uniform(*PREFIX_LEN_RANGE))
        n_shots = int(rng.randint(2, 11))            # paper: two to ten
        parts = []
        qa_texts = []
        for _ in range(n_shots):
            ex = train[int(shot_pool[sp % len(shot_pool)])]
            sp += 1
            if dataset == "rte":
                qa_texts.append(rte_shot(ex))
            else:
                qa_texts.append(copa_context(ex)
                                + copa_cont(ex["choice1"] if ex["label"] == 0
                                            else ex["choice2"]) + ".")
        qa_tok = sum(len(tokenizer(t, add_special_tokens=False).input_ids)
                     for t in qa_texts)
        ctx_budget = max(200, target_len - qa_tok - 4 * n_shots)
        for k in range(n_shots):
            passage = cursor.take(tokenizer, ctx_budget // n_shots)
            parts.append(f"Context: {passage}\n{qa_texts[k]}")
        ids = tokenizer("\n\n".join(parts)).input_ids[:PREFIX_CAP]
        prefixes.append(ids)
        used_bytes += len(ids) * BYTES_PER_TOKEN

    # ---- questions (validation; paper metric = accuracy)
    n_q = min(cfg["num_questions"], len(val))
    q_idx = rng.choice(len(val), n_q, replace=False)
    items = []
    for i in q_idx:
        ex = val[int(i)]
        if dataset == "rte":
            from flexllmgen.impress.verify_selection_rte import format_rte
            q = tokenizer("\n\n" + format_rte(ex),
                          add_special_tokens=False).input_ids
            items.append(dict(q=q, label=ex["label"]))
        else:
            base = "\n\n" + copa_context(ex)
            chs = [tokenizer(base + copa_cont(ex[f"choice{c}"]),
                             add_special_tokens=False).input_ids
                   for c in (1, 2)]
            cont_lens = [len(tokenizer(copa_cont(ex[f"choice{c}"]),
                                       add_special_tokens=False).input_ids)
                         for c in (1, 2)]
            items.append(dict(chs=chs, cont_lens=cont_lens,
                              label=int(ex["label"])))

    # ---- reuse frequency: normal distribution over the pool (paper)
    P = len(prefixes)
    w = np.exp(-0.5 * ((np.arange(P) - (P - 1) / 2) / max(P / 4, 1)) ** 2)
    w /= w.sum()
    assign = rng.choice(P, n_q, p=w).tolist()
    return prefixes, items, assign


def build_natural_prefixes(dataset, tokenizer, seed, n_pref=8):
    """Unextended prefixes for the accuracy-vs-retention experiment (a).
    Paper reading: §6.1 describes the accuracy protocol FIRST and only then
    says "Additionally, to test TTFT with long prefixes ..., we extend the
    prefixes" — i.e. the 4.8-5.7k extension belongs to the TTFT experiment;
    generation-quality runs use the plain 2-10 few-shot prompts."""
    from datasets import load_dataset
    rng = np.random.RandomState(seed + 1)
    if dataset == "rte":
        d = load_dataset("glue", "rte")
    else:
        d = load_dataset("super_glue", "copa", trust_remote_code=True)
    train = d["train"]
    order = rng.permutation(len(train)).tolist()
    sp = 0
    prefixes = []
    for _ in range(n_pref):
        n_shots = int(rng.randint(2, 11))
        shots = []
        for _ in range(n_shots):
            ex = train[int(order[sp % len(order)])]
            sp += 1
            if dataset == "rte":
                shots.append(rte_shot(ex))
            else:
                shots.append(copa_context(ex)
                             + copa_cont(ex["choice1"] if ex["label"] == 0
                                         else ex["choice2"]) + ".")
        prefixes.append(tokenizer("\n\n".join(shots)).input_ids)
    return prefixes


# ------------------------------------------------------------- serving
def serve_and_score(server, dataset, prefixes, items, assign, tokenizer,
                    time_it=False):
    """One pass over all questions; returns (accuracy, ttft_list)."""
    ctrl = server.controller
    true_id = tokenizer(" True", add_special_tokens=False).input_ids[0]
    false_id = tokenizer(" False", add_special_tokens=False).input_ids[0]
    preds, ttfts = [], []
    for it, p_idx in zip(items, assign):
        p_ids = prefixes[p_idx]
        if dataset == "rte":
            logits, ttft, _, _ = server.request(p_ids, it["q"])
            preds.append(0 if logits[true_id] > logits[false_id] else 1)
            ttfts.append(ttft)
        else:
            scores = []
            for c in range(2):
                ctrl.capture_full_logits = True
                _, ttft, _, _ = server.request(p_ids, it["chs"][c])
                ttfts.append(ttft)
                lp = torch.log_softmax(ctrl.full_logits[0], dim=-1)
                n, L = len(it["chs"][c]), it["cont_lens"][c]
                scores.append(float(sum(lp[n - L - 1 + k, it["chs"][c][n - L + k]]
                                        for k in range(L))))
            preds.append(int(np.argmax(scores)))
    acc = float(np.mean([p == it["label"]
                         for p, it in zip(preds, items)]))
    return acc, ttfts


def dense_score(model, ctrl, dataset, prefixes, items, assign, tokenizer):
    """ReComp: full dense prefill of prefix+query (+choice)."""
    true_id = tokenizer(" True", add_special_tokens=False).input_ids[0]
    false_id = tokenizer(" False", add_special_tokens=False).input_ids[0]
    preds, ttfts = [], []
    for it, p_idx in zip(items, assign):
        p_ids = prefixes[p_idx]
        if dataset == "rte":
            t0 = time.perf_counter()
            model.generate((list(p_ids) + list(it["q"]),), max_new_tokens=1,
                           do_sample=False)
            ttfts.append(time.perf_counter() - t0)
            lg = ctrl.last_logits[0]
            preds.append(0 if lg[true_id] > lg[false_id] else 1)
        else:
            scores = []
            for c in range(2):
                ctrl.capture_full_logits = True
                ids = list(p_ids) + list(it["chs"][c])
                t0 = time.perf_counter()
                model.generate((ids,), max_new_tokens=1, do_sample=False)
                ttfts.append(time.perf_counter() - t0)
                lp = torch.log_softmax(ctrl.full_logits[0], dim=-1)
                n, L = len(ids), it["cont_lens"][c]
                scores.append(float(sum(
                    lp[n - L - 1 + k, ids[n - L + k]] for k in range(L))))
            preds.append(int(np.argmax(scores)))
    acc = float(np.mean([p == it["label"]
                         for p, it in zip(preds, items)]))
    return acc, ttfts


def _dir_gb(path):
    total = 0
    for dp, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(dp, f))
    return total / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("rte", "copa"), required=True)
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--path", type=str, default="~/opt_weights")
    parser.add_argument("--offload-dir", type=str,
                        default="~/flexllmgen_offload_dir")
    parser.add_argument("--tree-root", type=str,
                        default="/home/dblab/hj/impress_kv_store")
    parser.add_argument("--gpu-cache-gb", type=float, default=10.0,
        help="paper value 10GB; on 4090 OOM re-run with the computed max")
    parser.add_argument("--cpu-cache-gb", type=float, default=32.0)
    parser.add_argument("--keep-tree", action="store_true",
        help="reuse an already-built pool (skip inserts)")
    parser.add_argument("--phases", type=str, default="a,b")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    phases = {p.strip() for p in args.phases.split(",")}
    cfg = DATASET_CFG[args.dataset]

    from flexllmgen.impress.pos_extend import install_pos_extension
    install_pos_extension(PREFIX_CAP)
    print(f"position extension: PI to {PREFIX_CAP} (paper 10K cap)")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b",
                                              padding_side="left")
    prefixes, items, assign = build_workload(args.dataset, tokenizer,
                                             args.seed)
    lens = [len(p) for p in prefixes]
    pool_gb_planned = sum(lens) * BYTES_PER_TOKEN / 1e9
    print(f"[{args.dataset}] pool: {len(prefixes)} prefixes, "
          f"len {min(lens)}-{max(lens)} (mean {np.mean(lens):.0f}) tokens, "
          f"planned KV {pool_gb_planned:.1f} GB (paper target "
          f"{cfg['pool_gb']:.0f} GB); {len(items)} questions; "
          f"shots/prefix in [2,10]")

    # ---------------------------------------------------------- model
    from flexllmgen.compression import CompressionConfig
    from flexllmgen.flex_opt import Policy, OptLM, get_opt_config
    from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk,
        TorchMixedDevice)
    from flexllmgen.utils import ExecutionEnv
    config = get_opt_config(args.model)
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk,
                       mixed=TorchMixedDevice([gpu, cpu, disk]))
    policy = Policy(1, 1, 100, 0, 100, 0, 100, 0,
                    overlap=False, sep_layer=True, pin_weight=True,
                    cpu_cache_compute=False, attn_sparsity=1.0,
                    compress_weight=False,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64, group_dim=0, symmetric=False),
                    compress_cache=False,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64, group_dim=2, symmetric=False))
    print(f"init weights: {args.model} ...")
    model = OptLM(config, env, args.path, policy)

    bpv = config.input_dim * 2
    icfg = ImpressConfig(
        retention_ratio=cfg["retention_ttft"],
        gpu_cache_capacity=int(args.gpu_cache_gb * 1e9 / bpv),
        cpu_cache_capacity=int(args.cpu_cache_gb * 1e9 / bpv),
        reorder_every_n_requests=None)   # manual pass between windows
    tree_dir = os.path.join(args.tree_root, f"paper_{args.dataset}")
    if not args.keep_tree:
        shutil.rmtree(tree_dir, ignore_errors=True)
    server = ImpressServer(model, config, tree_dir, icfg,
                           gpu_device=gpu.dev)
    ctrl = server.controller
    cache_pct = (args.gpu_cache_gb + args.cpu_cache_gb) / cfg["pool_gb"]
    print(f"caches: GPU {args.gpu_cache_gb:.0f} GB + CPU "
          f"{args.cpu_cache_gb:.0f} GB = {cache_pct:.0%} of pool "
          f"(paper 42GB/{cfg['pool_gb']:.0f}GB = "
          f"{42 / cfg['pool_gb']:.0%})")

    results = {}
    try:
        # ------------------------------------------------ pool build
        if not args.keep_tree:
            t0 = time.perf_counter()
            stub = tokenizer("\n\nAnswer:", add_special_tokens=False).input_ids
            for i, p_ids in enumerate(prefixes):
                server.request(p_ids, stub)
                if (i + 1) % 5 == 0:
                    print(f"  built {i + 1}/{len(prefixes)} prefixes ...")
            print(f"pool built: {_dir_gb(server.tree.nodes_dir):.1f} GB on "
                  f"disk ({time.perf_counter() - t0:.0f}s)")
        else:
            for p_ids in prefixes:
                assert not server.tree.match(p_ids).NR, "tree incomplete"
            print(f"pool reused: {_dir_gb(server.tree.nodes_dir):.1f} GB")

        # ------------------------------------ warm-up + one reorder pass
        t0 = time.perf_counter()
        serve_and_score(server, args.dataset, prefixes, items, assign,
                        tokenizer)
        done = server.reorderer.reorder_all()
        print(f"warm-up + reorder pass: {len(done)} nodes reordered "
              f"({time.perf_counter() - t0:.0f}s)")

        # --------------------------- (a) accuracy vs retention (Fig 4/15)
        # runs on NATURAL (unextended) 2-10-shot prefixes — the paper
        # extends prefixes only "to test TTFT"; with extended prefixes all
        # modes sit in interpolated-position territory and base quality
        # collapses (measured: ReComp ~= chance), inverting the curve.
        if "a" in phases:
            nat_prefixes = build_natural_prefixes(args.dataset, tokenizer,
                                                  args.seed)
            nlens = [len(p) for p in nat_prefixes]
            rng_a = np.random.RandomState(args.seed + 2)
            P = len(nat_prefixes)
            w = np.exp(-0.5 * ((np.arange(P) - (P - 1) / 2)
                               / max(P / 4, 1)) ** 2)
            w /= w.sum()
            nat_assign = rng_a.choice(P, len(items), p=w).tolist()
            print(f"\n(a) accuracy vs retention — natural 2-10-shot "
                  f"prefixes ({min(nlens)}-{max(nlens)} tokens), "
                  f"{len(items)} questions each:")
            stub = tokenizer("\n\nAnswer:",
                             add_special_tokens=False).input_ids
            for p_ids in nat_prefixes:
                server.request(p_ids, stub)
            serve_and_score(server, args.dataset, nat_prefixes, items,
                            nat_assign, tokenizer)   # warm-up
            server.reorderer.reorder_all()
            accs = {}
            for r in RETENTION_SWEEP:
                ctrl.retention_ratio = r
                acc, _ = serve_and_score(server, args.dataset, nat_prefixes,
                                         items, nat_assign, tokenizer)
                accs[r] = acc
                print(f"  retention {r:4.0%}: accuracy {acc * 100:5.1f}%")
            ctrl.enabled = False
            acc_ref, _ = dense_score(model, ctrl, args.dataset,
                                     nat_prefixes, items, nat_assign,
                                     tokenizer)
            print(f"  ReComp reference       : accuracy {acc_ref*100:5.1f}%")
            results["retention_acc"] = accs
            results["retention_ref"] = acc_ref

        # ------------------------------ (b) TTFT, 3 modes (Fig 16/17)
        if "b" in phases:
            r = cfg["retention_ttft"]
            print(f"\n(b) TTFT at paper retention ({r:.0%}):")
            ctrl.retention_ratio = r
            server.manager.phys.reset()
            ctrl.io.reset()
            acc_i, tt_i = serve_and_score(server, args.dataset, prefixes,
                                          items, assign, tokenizer,
                                          time_it=True)
            phys = server.manager.phys.summary()
            n_req = len(tt_i)
            disk_i = phys["disk_ms"] / n_req
            print(f"  IMPRESS : acc {acc_i*100:5.1f}%, TTFT "
                  f"{np.median(tt_i)*1e3:7.1f} ms (disk {disk_i:.1f} ms, "
                  f"pcie {phys['pcie_mb']/n_req:.0f} MB/req; fallback "
                  f"{ctrl.io.fallback_rate()*100:.1f}%)")

            # free the GPU tier for dense/FullLoad transients
            for key, meta in list(server.manager.cache.chunks.items()):
                if meta.location == "gpu":
                    server.manager.cache.drop(key)
            torch.cuda.empty_cache()

            fl_mgr = ChunkKVManager(
                server.tree, gpu.dev,
                ImpressConfig(retention_ratio=1.0, gpu_cache_capacity=0,
                              cpu_cache_capacity=0),
                n_head=config.n_head)
            ctrl.retention_ratio = 1.0
            old_provider_hook = None
            preds_ttfts = []
            # FullLoad: serve via zero-cache provider
            ctrl.selected_by_layer = None
            ctrl.token_importance = None

            def fullload_request(p_ids, q_ids):
                t0 = time.perf_counter()
                m = server.tree.match(p_ids)
                ctrl.provider = PathKVProvider(fl_mgr, m)
                ctrl.prefix_len = m.r_len
                ctrl.enabled = True
                model.generate((list(q_ids),), max_new_tokens=1,
                               do_sample=False)
                dt = time.perf_counter() - t0
                ctrl.enabled = False
                ctrl.provider = None
                return dt

            true_id = tokenizer(" True", add_special_tokens=False).input_ids[0]
            false_id = tokenizer(" False",
                                 add_special_tokens=False).input_ids[0]
            preds_f, tt_f = [], []
            for it, p_idx in zip(items, assign):
                p_ids = prefixes[p_idx]
                if args.dataset == "rte":
                    tt_f.append(fullload_request(p_ids, it["q"]))
                    lg = ctrl.last_logits[0]
                    preds_f.append(0 if lg[true_id] > lg[false_id] else 1)
                else:
                    scores = []
                    for c in range(2):
                        ctrl.capture_full_logits = True
                        tt_f.append(fullload_request(p_ids, it["chs"][c]))
                        lp = torch.log_softmax(ctrl.full_logits[0], dim=-1)
                        n, L = len(it["chs"][c]), it["cont_lens"][c]
                        scores.append(float(sum(
                            lp[n - L - 1 + k, it["chs"][c][n - L + k]]
                            for k in range(L))))
                    preds_f.append(int(np.argmax(scores)))
            acc_f = float(np.mean([p == it["label"]
                                   for p, it in zip(preds_f, items)]))
            disk_f = fl_mgr.phys.summary()["disk_ms"] / len(tt_f)
            print(f"  FullLoad: acc {acc_f*100:5.1f}%, TTFT "
                  f"{np.median(tt_f)*1e3:7.1f} ms (disk {disk_f:.1f} ms)")

            ctrl.enabled = False
            acc_r, tt_r = dense_score(model, ctrl, args.dataset, prefixes,
                                      items, assign, tokenizer)
            print(f"  ReComp  : acc {acc_r*100:5.1f}%, TTFT "
                  f"{np.median(tt_r)*1e3:7.1f} ms")

            ti, tf, tr = (np.median(tt_i), np.median(tt_f), np.median(tt_r))
            io_ratio = disk_f / max(disk_i, 1e-9)
            print(f"\n  === paper-band check ({args.dataset}, retention "
                  f"{r:.0%}) ===")
            print(f"  prefix-KV I/O time reduction vs FullLoad: "
                  f"{io_ratio:.2f}x  (paper: 1.5-3.8x)")
            print(f"  TTFT improvement: vs FullLoad {tf/ti:.2f}x, "
                  f"vs ReComp {tr/ti:.2f}x  (paper: 1.2-2.8x)")
            print(f"  accuracy: IMPRESS {acc_i*100:.1f}% vs ReComp "
                  f"{acc_r*100:.1f}% (diff {(acc_i-acc_r)*100:+.1f}%p; "
                  f"paper: <1%p at this retention)")
            results["ttft"] = dict(impress=ti, fullload=tf, recomp=tr,
                                   disk_i=disk_i, disk_f=disk_f,
                                   acc=(acc_i, acc_f, acc_r))
        print("DONE")
    except torch.OutOfMemoryError as e:
        import traceback
        traceback.print_exc()
        print(f"[allocated {torch.cuda.memory_allocated()/1e9:.1f} GB, "
              f"reserved {torch.cuda.memory_reserved()/1e9:.1f} GB]")
        free_gb = (23.64 - 13.34 - 2.3)
        print(f"\nOOM with GPU cache {args.gpu_cache_gb:.0f} GB (paper "
              f"value assumes an 80GB A100). Computed 4090 maximum: "
              f"23.64 total - 13.34 weights - ~2.3 runtime ~= "
              f"{free_gb:.1f} GB.")
        print(f"Re-run with: --gpu-cache-gb 8 --keep-tree")
        sys.exit(3)
    finally:
        server.close()
        env.close_copy_threads()


if __name__ == "__main__":
    main()
