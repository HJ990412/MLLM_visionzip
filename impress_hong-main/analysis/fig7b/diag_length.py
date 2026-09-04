"""Diagnostic only (not the reported result): input-length sensitivity of
the mean head-pair Jaccard at the middle layer (L16), top 50%."""
import sys
sys.path.insert(0, "/home/dblab/hj/FlexGen")
sys.path.insert(0, "/home/dblab/hj/analysis/fig7b")
import numpy as np, torch
from fig7b_jaccard import topk_sets, jaccard_matrix, offdiag_mean

from transformers import AutoTokenizer
from flexllmgen.impress.verify_impress_e2e import build_workload
tok = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
prefix_texts, _ = build_workload(4, 21, 0)
full = tok(prefix_texts[0]).input_ids

from flexllmgen.compression import CompressionConfig
from flexllmgen.flex_opt import Policy, OptLM, get_opt_config
from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
from flexllmgen.utils import ExecutionEnv
from flexllmgen.impress.importance import install_importance_hook, uninstall_importance_hook

config = get_opt_config("facebook/opt-6.7b")
gpu, cpu = TorchDevice("cuda:0"), TorchDevice("cpu")
disk = TorchDisk("~/flexllmgen_offload_dir")
env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk,
                   mixed=TorchMixedDevice([gpu, cpu, disk]))
policy = Policy(1, 1, 100, 0, 100, 0, 100, 0, overlap=False, sep_layer=True,
                pin_weight=True, cpu_cache_compute=False, attn_sparsity=1.0,
                compress_weight=False,
                comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
                compress_cache=False,
                comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False))
print("init weights ...")
model = OptLM(config, env, "~/opt_weights", policy)
col = install_importance_hook()
try:
    print(f"{'len':>6s} {'L16 r50':>8s} {'L16 r10':>8s} {'L1 r50':>7s} {'best layer(r50)':>16s}")
    for n in (32, 64, 128, 256, 512, 1642):
        col.clear(); col.enabled = True
        model.generate((full[:n],), max_new_tokens=1, do_sample=False)
        col.enabled = False
        vals = {}
        for j in (0, 15):
            sets, _ = topk_sets(col.scores[j][0].float(), 0.50)
            vals[j] = offdiag_mean(jaccard_matrix(sets))
        sets10, _ = topk_sets(col.scores[15][0].float(), 0.10)
        v10 = offdiag_mean(jaccard_matrix(sets10))
        best = max(range(32), key=lambda j: offdiag_mean(jaccard_matrix(
            topk_sets(col.scores[j][0].float(), 0.50)[0])))
        bestv = offdiag_mean(jaccard_matrix(
            topk_sets(col.scores[best][0].float(), 0.50)[0]))
        print(f"{n:>6d} {vals[15]:8.3f} {v10:8.3f} {vals[0]:7.3f} "
              f"L{best+1}={bestv:.3f}")
finally:
    uninstall_importance_hook()
    env.close_copy_threads()
