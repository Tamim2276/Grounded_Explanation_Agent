# Download everything the 10 days need, in one go, while there is power and internet
# (BUILD_PLAN.md, Day 1). After this, nothing in the plan needs the internet.
#
#   python scripts/download_all.py --dry-run   # show what would be downloaded, and the sizes
#   python scripts/download_all.py             # download (~16 GB, mostly ColQwen2 and Qwen2.5-VL)
#
# Safe to stop at any moment -- Ctrl+C or a power cut -- and run again: finished
# files are skipped, Hugging Face resumes half-downloaded files, and a PDF only gets
# its real name once it is complete and opens. Smallest and most important first, so
# even a short power window makes progress.
import argparse
import json
import sys
import time
from pathlib import Path

import pymupdf
import requests
from huggingface_hub import HfApi, constants, hf_hub_download, snapshot_download

from gea import config as cfg
from gea.safeio import atomic_path, remove_leftovers

SPIQA_REPO = "google/spiqa"
SPIQA_FILES = ["test-A/SPIQA_testA.json", "test-A/SPIQA_testA_Images.zip"]

HF_MODELS = [                                    # (repo, what it is for)
    ("allenai/specter2_base", "SPECTER2, the text index (Day 3)"),
    ("allenai/specter2", "SPECTER2 adapter for paragraphs"),
    ("allenai/specter2_adhoc_query", "SPECTER2 adapter for questions"),
    ("microsoft/table-structure-recognition-v1.1-all", "Table Transformer, table cells (Day 4)"),
    ("vidore/colqwen2-v1.0", "ColQwen2 retrieval adapter (Days 3-4)"),
    ("vidore/colqwen2-base", "ColQwen2 base weights (8 GB)"),
]

GGUF_REPO = "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF"
GGUF_FILES = ["mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf",      # the vision part
              "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"]          # the language model, 4-bit
GGUF_DIR = cfg.MODELS_DIR / "qwen2.5-vl-7b"                  # scripts/start_llm.ps1 looks here

ARXIV_DELAY = 3.0                                # seconds between requests (arXiv's rule)


def spiqa(data_dir: Path, dry: bool) -> None:
    for name in SPIQA_FILES:
        print(f"  SPIQA {name}")
        if not dry:
            hf_hub_download(SPIQA_REPO, name, repo_type="dataset", local_dir=data_dir / "spiqa")


def paper_ids(data_dir: Path) -> list:
    path = data_dir / "spiqa" / SPIQA_FILES[0]
    return list(json.load(open(path, encoding="utf-8"))) if path.exists() else []


def pdfs(data_dir: Path, dry: bool, limit: int = None) -> list:
    # One PDF per SPIQA test-A paper, from arXiv. Returns the ids that failed.
    ids = paper_ids(data_dir)[:limit]
    if not ids:
        print("  arXiv PDFs: the list comes from SPIQA_testA.json (not downloaded yet)")
        return []
    folder = data_dir / "pdfs"
    todo = [pid for pid in ids if not (folder / f"{pid.replace('/', '_')}.pdf").exists()]
    print(f"  arXiv PDFs: {len(ids) - len(todo)} of {len(ids)} already here, {len(todo)} to fetch")
    if dry:
        return []
    failed = []
    for n, pid in enumerate(todo, 1):
        out = folder / f"{pid.replace('/', '_')}.pdf"
        try:
            reply = requests.get(f"https://arxiv.org/pdf/{pid}", timeout=60,
                                 headers={"User-Agent": "thesis-research (Grounded_Explanation_Agent)"})
            reply.raise_for_status()
            with atomic_path(out) as tmp:
                tmp.write_bytes(reply.content)
                with pymupdf.open(tmp) as doc:            # a PDF that does not open never gets its name
                    pages = doc.page_count
            print(f"    [{n}/{len(todo)}] {pid}  {pages} pages")
        except Exception as e:                            # network down, refused, not a PDF
            failed.append(pid)
            print(f"    [{n}/{len(todo)}] {pid}  FAILED ({type(e).__name__}) -- run the script again later")
        time.sleep(ARXIV_DELAY)
    return failed


def models(dry: bool, api: HfApi) -> None:
    for repo, why in HF_MODELS:
        if dry:
            size = sum(s.size or 0 for s in api.repo_info(repo, files_metadata=True).siblings)
            print(f"  {repo:50s} {size / 2**30:5.2f} GB  {why}")
        else:
            print(f"  {repo}  ({why})")
            snapshot_download(repo)


def gguf(dry: bool, api: HfApi) -> None:
    for name in GGUF_FILES:
        if dry:
            info = api.get_paths_info(GGUF_REPO, [name])[0]
            print(f"  {name:50s} {info.size / 2**30:5.2f} GB  -> {GGUF_DIR}")
        else:
            print(f"  {name}")
            hf_hub_download(GGUF_REPO, name, local_dir=GGUF_DIR)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="only show what would be downloaded")
    ap.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    ap.add_argument("--limit-pdfs", type=int, default=None, help="fetch only the first N PDFs (for testing)")
    ap.add_argument("--allow-c-drive", action="store_true", help="download even if the cache is on C:")
    args = ap.parse_args()

    cache = Path(constants.HF_HUB_CACHE).resolve()
    print(f"Hugging Face cache : {cache}")
    print(f"data folder        : {args.data_dir.resolve()}")
    if cache.drive.upper() == "C:" and not args.allow_c_drive:
        print("STOP: the model cache is on C:, which is nearly full. Set HF_HOME to a folder on D:\n"
              "      (BUILD_PLAN.md, Day 1, step 1.2), open a NEW terminal, and run this again.")
        return 1

    removed = remove_leftovers(args.data_dir)
    if removed:
        print(f"removed {len(removed)} half-written file(s) left by a power cut")

    api = HfApi()
    print("\n1. SPIQA test-A (questions + reference images)")
    spiqa(args.data_dir, args.dry_run)
    print("\n2. The papers")
    failed = pdfs(args.data_dir, args.dry_run, args.limit_pdfs)
    print("\n3. Retrieval and table models (Hugging Face cache)")
    models(args.dry_run, api)
    print("\n4. The language model for llama.cpp")
    gguf(args.dry_run, api)

    if failed:
        print(f"\n{len(failed)} PDF(s) failed: {failed}\nRun the script again; finished files are skipped.")
        return 1
    print("\nDry run only -- nothing was downloaded." if args.dry_run else "\nAll downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
