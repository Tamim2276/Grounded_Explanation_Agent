# N4's two indexes (BUILD_PLAN.md Day 3). This file starts with the SPECTER2 text
# encoder (step 3.2); the text index (3.3) and the ColQwen2 page index (3.4) build on it.
from functools import lru_cache

import numpy as np
import torch

from gea import config as cfg

SPECTER2_BASE = "allenai/specter2_base"
# One base model, two small add-ons ("adapters"), one per side of the search:
# kind -> (repository, name the adapter is loaded as)
SPECTER2_ADAPTERS = {"doc":   ("allenai/specter2", "proximity"),              # paragraphs, captions
                     "query": ("allenai/specter2_adhoc_query", "adhoc_query")}  # questions
BATCH = 16           # texts per forward pass
MAX_TOKENS = 512     # SPECTER2's limit; longer texts are cut (a 150-word chunk is ~200 tokens)


def local_snapshot(repo: str) -> str:
    # The folder scripts/download_all.py downloaded the model into. Never the internet:
    # a power cut often takes the internet with it, and a missing model should say so
    # at once instead of hanging on a download.
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError
    try:
        return snapshot_download(repo, local_files_only=True)
    except LocalEntryNotFoundError:
        raise FileNotFoundError(f"{repo} is not downloaded -- run: python scripts/download_all.py") from None


@lru_cache(maxsize=1)
def specter():
    # SPECTER2 and both adapters, loaded once per session (about 4 s). It stays on the
    # CPU: it is small (110M weights), and the B580 is kept for ColQwen2 and the
    # language model. Imported here, not at the top, so the stub backend never pays
    # for importing transformers.
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    base = local_snapshot(SPECTER2_BASE)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoAdapterModel.from_pretrained(base)
    for repo, name in SPECTER2_ADAPTERS.values():
        model.load_adapter(local_snapshot(repo), load_as=name)
    return tok, model.eval()


@torch.inference_mode()
def embed_text(texts: list, kind: str = "doc") -> np.ndarray:
    # One SPECTER2 vector per text, shape (len(texts), 768), each scaled to length 1,
    # so the dot product of two vectors is their cosine similarity (1 = same meaning).
    # kind="doc" for paragraphs and captions, kind="query" for questions.
    if kind not in SPECTER2_ADAPTERS:
        raise ValueError(f"kind must be 'doc' or 'query', not {kind!r}")
    tok, model = specter()
    model.set_active_adapters(SPECTER2_ADAPTERS[kind][1])
    out = [np.zeros((0, cfg.SPECTER2_DIM), dtype=np.float32)]
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=MAX_TOKENS,
                  return_tensors="pt")
        cls = model(**enc).last_hidden_state[:, 0]     # the first ([CLS]) slot: SPECTER2's summary
        out.append(torch.nn.functional.normalize(cls, dim=-1).numpy())
    return np.concatenate(out).astype(np.float32)      # FAISS wants float32
