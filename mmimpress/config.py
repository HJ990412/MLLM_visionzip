"""Global constants for the image-prefix IMPRESS (SparseMM importance) system.

Model: llava-hf/llava-v1.6-vicuna-7b-hf (LLaVA-NeXT, Vicuna-7B backbone).
It is a *MHA* model (32 layers x 32 query heads == 32 KV heads), which is a
deliberate choice: SparseMM's per-head asymmetric KV budget only converts into
real disk-byte savings when each head owns its own K/V.  Under grouped-query
attention (e.g. Qwen2.5-VL: 28 q-heads / 4 kv-heads) a KV group can only be
skipped when all 7 query heads sharing it are non-visual, which mostly never
happens.  See docs/DESIGN.md.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
LOAD_4BIT = True
ATTN_IMPL = "eager"          # required: sdpa/flash never materialise weights

STORE_DTYPE = "float16"      # on-disk prefix KV dtype
COMPUTE_DTYPE = "bfloat16"   # dtype fed back into the model

# ---- storage layout -------------------------------------------------------
# HEAD-MAJOR: layer_{l}/{k,v}.bin holds (num_heads, v_num, head_dim) fp16, so
# one head's whole visual K (or V) is a single contiguous byte range and a
# chunk is a contiguous sub-range inside it.  This is the change that makes
# per-head skipping physically realisable; a token-major (v_num, H, hd) store
# interleaves heads and forces every read to touch every head.
CHUNK_SIZE = 64              # tokens per chunk *within one head* (IMPRESS 6.1)

# ---- SparseMM budget ------------------------------------------------------
RETENTION_RATIO = 0.25       # overall visual-KV budget (IMPRESS retention)
BASE_FRACTION = 0.20         # share of the budget given uniformly to all heads
                             # (SparseMM "uniform base"); the rest is handed
                             # out in proportion to the visual-head score.
LOCAL_WINDOW = 0             # trailing visual tokens always kept (0 = off;
                             # LLaVA's newline separators are kept instead)

# ---- calibration / scoring ------------------------------------------------
PROBE_HEADS = 3              # IMPRESS 4.3: probe heads per layer
ALPHA = 0.6                  # IMPRESS 4.3: similarity threshold t = j ** alpha
OBS_WINDOW = 32              # trailing prompt rows used as the observation
                             # window for per-token attention accumulation
CALIB_SAMPLES = 64           # questions used to score visual heads offline

# ---- paths ----------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
STORE_DIR = PROJECT_ROOT / "kvstore"
RESULTS_DIR = PROJECT_ROOT / "results"
