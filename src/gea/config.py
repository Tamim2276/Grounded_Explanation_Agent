# Every constant and every switch of the system, in one place.
#
# Read a setting as cfg.NAME at the moment you need it -- never with
# `from gea.config import NAME`. The notebook and the tests change settings while
# the kernel runs, and a name imported by value would never see the change.
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # src/gea/config.py -> the repository
DATA_DIR = PROJECT_ROOT / "data"                     # big, regenerable files; not in git
MODELS_DIR = DATA_DIR / "models"                     # the GGUF language model (scripts/download_all.py)
# Every model reply, one small JSON file each (BUILD_PLAN.md Day 5). It lives in git, not
# in data/, so every checkpoint also backs up the GPU hours that produced the results.
LLM_CACHE_DIR = PROJECT_ROOT / "results" / "llm_cache"

# --- what the agent can retrieve --------------------------------------------
MODALITIES = ("text", "table", "figure")

# --- loop bounds and retrieval ------------------------------------------------
MAX_ROUNDS   = 3              # bound on the retrieval loop   (G1 -> replan_prep -> N5)
MAX_REWRITES = 2              # bound on the regenerate loop  (G2 -> rewrite_prep -> N9)
TOP_K        = 1              # items each retriever returns per call
ZOOM_IF_AREA_BELOW = 0.30     # N7 runs when a figure covers less than 30% of its page

# --- the real text search (N6a on the real backend, BUILD_PLAN.md Day 3) ------
TEXT_SEARCH = "hybrid"        # how paragraphs are ranked: "specter2" | "keyword" | "hybrid"
                              # (dev: results/day3_text_search.json, notes/results_for_paper.md)
TEXT_TOP_K  = 3               # paragraphs per call: SPECTER2 ranks topic, not answer, so keep 3
RRF_K       = 60              # hybrid: a paragraph earns 1 / (RRF_K + its rank) from each search

# --- the real table and figure tools (N6b, N6c, N7 on the real backend, Day 4) -
REGION_SEARCH = "both"        # how a table/figure is chosen: "text" | "visual" | "both"
ZOOM_DPI      = 300           # N7 re-renders a figure at this resolution ...
ZOOM_MAX_PX   = 1024          # ... but no larger than this on its long side (image tokens)
SUB_GOAL_MODALITY = None      # testing knob for N1's stand-in: force every sub-goal's
                              # modality ("table" / "figure") so a tool can be exercised

# --- index shapes ---------------------------------------------------------------
SPECTER2_DIM = 768            # SPECTER2 embedding width (text index)
COLPALI_DIM  = 128            # ColQwen2 multi-vector width (page index)
PATCH_GRID   = (32, 32)       # stub patch grid per page (rows, cols); real ColQwen2 grids vary
PAGE_W, PAGE_H = 595.0, 842.0 # A4 page, in PDF points (the fake paper's page size)

SEED = 42

# --- switches -------------------------------------------------------------------
BACKEND      = "stub"         # "stub": the fake paper; "real": SPIQA papers (BUILD_PLAN.md Day 2 on)
PLANNER_MODE = "rules"        # N5: "rules" (stand-in) or "llm" (the local model)
TRACE        = True           # print one line per node as the graph runs

# --- knobs for the generator stand-in (stub backend only) -----------------------
HALLUCINATE_ONCE   = False    # the first draft misquotes one number, later drafts are honest
HALLUCINATE_ALWAYS = False    # every draft misquotes it -- exercises the rewrite budget
TABLE_CLAIM_STYLE  = "honest" # "honest" | "misattributed" | "uncited" (notebook Section 15)

# --- the local language model (llama.cpp llama-server, BUILD_PLAN.md Day 5) -----
LLM_URL   = "http://localhost:8080/v1"
LLM_MODEL = "qwen2.5-vl-7b"   # informational: the server answers with what it loaded


@contextmanager
def override(**settings):
    # Change settings for the length of a `with` block, then put them back -- even
    # if the block crashes. Forgetting to reset a knob by hand would silently put
    # every later run in the wrong situation.
    #
    #     with cfg.override(TRACE=False, HALLUCINATE_ONCE=True):
    #         out = agent.invoke(new_state(QUERY))
    unknown = sorted(k for k in settings if k not in globals())
    if unknown:
        raise AttributeError(f"unknown setting(s): {unknown}")
    saved = {k: globals()[k] for k in settings}
    globals().update(settings)
    try:
        yield
    finally:
        globals().update(saved)
