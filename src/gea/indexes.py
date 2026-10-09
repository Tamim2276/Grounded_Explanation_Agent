# N4's two indexes (BUILD_PLAN.md Day 3): the SPECTER2 text encoder (step 3.2), the
# text index built from it (3.3), and the ColQwen2 page index (3.4).
import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import torch

from gea import config as cfg
from gea.safeio import atomic_path, atomic_write_json

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


@lru_cache(maxsize=2)
def specter(device: str = "cpu"):
    # SPECTER2 and both adapters, loaded once per session and device (about 4 s).
    # Questions are embedded on the CPU: SPECTER2 is small (110M weights), and the
    # B580 is kept for the language model. Building the index may use the B580
    # (device="xpu") when it is free. Imported here, not at the top, so the stub
    # backend never pays for importing transformers.
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    base = local_snapshot(SPECTER2_BASE)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoAdapterModel.from_pretrained(base)
    for repo, name in SPECTER2_ADAPTERS.values():
        model.load_adapter(local_snapshot(repo), load_as=name)
    return tok, model.to(device).eval()      # float32 on both devices: the same vectors


@torch.inference_mode()
def embed_text(texts: list, kind: str = "doc", device: str = "cpu") -> np.ndarray:
    # One SPECTER2 vector per text, shape (len(texts), 768), each scaled to length 1,
    # so the dot product of two vectors is their cosine similarity (1 = same meaning).
    # kind="doc" for paragraphs and captions, kind="query" for questions.
    if kind not in SPECTER2_ADAPTERS:
        raise ValueError(f"kind must be 'doc' or 'query', not {kind!r}")
    tok, model = specter(device)
    model.set_active_adapters(SPECTER2_ADAPTERS[kind][1])
    out = [np.zeros((0, cfg.SPECTER2_DIM), dtype=np.float32)]
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=MAX_TOKENS,
                  return_tensors="pt").to(device)
        cls = model(**enc).last_hidden_state[:, 0]     # the first ([CLS]) slot: SPECTER2's summary
        out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)      # FAISS wants float32


# --- the text index (step 3.3) ----------------------------------------------------
# Two files next to each paper's corpus.json:
TEXT_INDEX = "text.faiss"       # the vectors, one row per record
TEXT_IDS = "text_ids.json"      # the record behind each row: a chunk ("c3"), or the caption of "t1" / "f2"


def text_records(paper: dict) -> list:
    # (id, text) of everything the text index holds, in row order: every chunk, then
    # every table caption, then every figure caption.
    return ([(c["id"], c["text"]) for c in paper["chunks"]]
            + [(r["id"], r["caption"]) for r in paper["tables"] + paper["figures"]])


def is_current(folder: Path, name: str) -> bool:
    # The index file exists and was built after N3 last wrote the paper:
    # build_corpus.py --force makes an older index stale, because it would describe
    # the old records.
    index = Path(folder) / name
    return index.exists() and index.stat().st_mtime >= (Path(folder) / "corpus.json").stat().st_mtime


def text_index_is_current(folder: Path) -> bool:
    return is_current(folder, TEXT_INDEX)


def build_text_index(folder: Path, device: str = "cpu") -> int:
    # Embed every record of one paper with the "doc" adapter and save an exact
    # inner-product index; returns the number of rows. The ids are saved first and the
    # index last, so an existing index means a complete paper: a power cut in between
    # leaves no index, and the paper is simply built again.
    folder = Path(folder)
    paper = json.loads((folder / "corpus.json").read_text(encoding="utf-8"))
    ids, texts = zip(*text_records(paper))
    index = faiss.IndexFlatIP(cfg.SPECTER2_DIM)             # compare with every row; IP = inner product,
    index.add(embed_text(list(texts), "doc", device))       # = cosine similarity for length-1 vectors
    atomic_write_json(folder / TEXT_IDS, {"model": f"{SPECTER2_BASE} + {SPECTER2_ADAPTERS['doc'][0]}",
                                          "ids": list(ids)})
    with atomic_path(folder / TEXT_INDEX) as tmp:
        faiss.write_index(index, str(tmp))
    return index.ntotal


def load_text_index(folder: Path) -> dict:
    folder = Path(folder)
    if not text_index_is_current(folder):
        raise FileNotFoundError(f"{folder.name}: no current text index -- run: python scripts/build_indexes.py text")
    index = faiss.read_index(str(folder / TEXT_INDEX))
    ids = json.loads((folder / TEXT_IDS).read_text(encoding="utf-8"))["ids"]
    if index.ntotal != len(ids):
        raise ValueError(f"{folder.name}: {index.ntotal} vectors but {len(ids)} ids "
                         f"-- run: python scripts/build_indexes.py text --force")
    return {"faiss": index, "ids": ids}


# --- the page index (step 3.4) ----------------------------------------------------
COLQWEN2_BASE = "vidore/colqwen2-base"
COLQWEN2 = "vidore/colqwen2-v1.0"      # a small LoRA add-on that turns the base into the retriever
PAGE_INDEX = "pages.pt"                 # next to corpus.json: every page's patch vectors and grid


@lru_cache(maxsize=1)
def colqwen(device: str = "cpu"):
    # ColQwen2: the base model with the v1.0 add-on merged in, in bfloat16 (2.2B weights,
    # about 4.5 GB). Page images are embedded on the B580; at question time a short
    # query can be embedded on the CPU, so the B580 stays free for the language model.
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from peft import PeftModel

    model = ColQwen2.from_pretrained(local_snapshot(COLQWEN2_BASE), torch_dtype=torch.bfloat16,
                                     low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, local_snapshot(COLQWEN2)).merge_and_unload()
    proc = ColQwen2Processor.from_pretrained(local_snapshot(COLQWEN2))
    return proc, model.to(device).eval()


def unload_models() -> None:
    # Drop every loaded model and hand the B580's memory back (to llama-server, or to
    # another experiment).
    import gc
    from gea.device import empty_cache
    specter.cache_clear()
    colqwen.cache_clear()
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        empty_cache(torch.device("xpu"))


@torch.inference_mode()
def embed_page(image, device: str = "cpu") -> tuple:
    # One page image -> (vectors, (rows, cols)). The model sees the page as a grid of
    # patches (about 31 x 24 for a 150-dpi page) and gives every patch its own
    # 128-number vector, stored row by row. Only the patch vectors are kept; the
    # prompt's text tokens around them are dropped.
    proc, model = colqwen(device)
    batch = proc.process_images([image]).to(device)
    out = model(**batch)[0]                                    # (tokens, 128), length 1 each
    patches = batch["input_ids"][0] == proc.image_token_id
    cols, rows = proc.get_n_patches(image.size, patch_size=model.patch_size,
                                    spatial_merge_size=model.spatial_merge_size)
    vectors = out[patches].cpu()                               # bfloat16, as the model made them
    if len(vectors) != rows * cols:
        raise ValueError(f"{len(vectors)} patch vectors for a {rows} x {cols} grid")
    return vectors, (rows, cols)


@torch.inference_mode()
def embed_page_query(question: str, device: str = "cpu") -> torch.Tensor:
    # A question -> one 128-number vector per query token, (tokens, 128).
    proc, model = colqwen(device)
    batch = proc.process_queries([question]).to(device)
    return model(**batch)[0].cpu()


def maxsim(query: torch.Tensor, pages: list) -> list:
    # The page score: for every query token, its best-matching patch on the page;
    # then the sum of those best matches. query (tokens, 128); pages: list of (patches, 128).
    q = query.float()
    return [float((q @ p.float().T).max(dim=1).values.sum()) for p in pages]


def page_index_is_current(folder: Path) -> bool:
    return is_current(folder, PAGE_INDEX)


def build_page_index(folder: Path, device: str = "xpu") -> int:
    # Embed every page image of one paper; returns the number of pages. One file per
    # paper, written whole at the end: a power cut costs only the paper in progress.
    from PIL import Image

    folder = Path(folder)
    paper = json.loads((folder / "corpus.json").read_text(encoding="utf-8"))
    pages = []
    for p in paper["pages"]:
        with Image.open(folder / p["image"]) as image:
            vectors, grid = embed_page(image.convert("RGB"), device)
        pages.append({"page": p["page"], "grid": grid, "vectors": vectors})
    with atomic_path(folder / PAGE_INDEX) as tmp:
        torch.save({"model": COLQWEN2, "pages": pages}, tmp)
    return len(pages)


def load_page_index(folder: Path) -> list:
    folder = Path(folder)
    if not page_index_is_current(folder):
        raise FileNotFoundError(f"{folder.name}: no current page index -- run: python scripts/build_indexes.py pages")
    return torch.load(folder / PAGE_INDEX, weights_only=True)["pages"]
