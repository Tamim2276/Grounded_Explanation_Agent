# Build N4's indexes for every paper scripts/build_corpus.py has read
# (BUILD_PLAN.md, Day 3, steps 3.3-3.4; the parsed tables, Day 4, step 4.3).
#
#   python scripts/build_indexes.py text          # SPECTER2 + FAISS on the CPU (~10 min)
#   python scripts/build_indexes.py text --gpu    # the same on the B580 (~1 min), when it is free
#   python scripts/build_indexes.py pages         # ColQwen2 page vectors, always on the B580
#   python scripts/build_indexes.py tables        # every table's cells (Table Transformer, CPU)
#   ... --force                                   # build every paper again
#
# Power-cut safe: a paper counts as done only when its index file exists, and that file
# is written last, so an interrupted paper is simply built again on the next run.
#
# Before loading a model the script checks that it fits: free memory (RAM + page file)
# and, for the B580, the GPU memory other programs already hold. Running out of either
# crashes not only this script but whatever else is running, such as another experiment.
import argparse
import shutil
import sys
import time

import torch

from gea import config as cfg
from gea.device import report_no_room, room_problems
from gea.indexes import (build_page_index, build_text_index, page_index_is_current,
                         text_index_is_current, unload_models)
from gea.tables import TABLES_DIR, build_table_cache, tables_are_current

CORPUS = cfg.DATA_DIR / "corpus"

# (index, device) -> (free memory it needs in GB, most GPU memory other programs may hold in MB)
NEEDS = {("text", "cpu"): (2, None),
         ("text", "xpu"): (3, 10_000),        # SPECTER2 takes ~1 GB of the 12 GB
         ("pages", "xpu"): (8, 4_500),        # ColQwen2 takes ~4.5 GB + working space; loads via RAM
         ("tables", "cpu"): (1.5, None)}      # the Table Transformer is small

JOBS = {"text": (build_text_index, text_index_is_current, "texts"),
        "pages": (build_page_index, page_index_is_current, "pages"),
        "tables": (build_table_cache, tables_are_current, "tables")}


def build(what: str, folders: list, device: str, force: bool) -> None:
    build_one, is_current, unit = JOBS[what]
    start, built, total = time.time(), 0, 0
    for n, folder in enumerate(folders, 1):
        if is_current(folder) and not force:
            continue
        if force and what == "tables":
            shutil.rmtree(folder / TABLES_DIR, ignore_errors=True)     # parse every table again
        t = time.time()
        count = build_one(folder, device)
        built, total = built + 1, total + count
        print(f"  [{n}/{len(folders)}] {folder.name}: {count} {unit} ({time.time() - t:.0f} s)", flush=True)
    print(f"{what} index: built {built} paper(s), {total} {unit}, in {(time.time() - start) / 60:.1f} min; "
          f"{len(folders) - built} were already current")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["text", "pages", "tables"], help="which index to build")
    ap.add_argument("--gpu", action="store_true", help="text index: use the B580 instead of the CPU")
    ap.add_argument("--force", action="store_true", help="build every paper again")
    ap.add_argument("--skip-checks", action="store_true", help="start even if memory looks short")
    args = ap.parse_args()

    device = "xpu" if args.what == "pages" or args.gpu else "cpu"
    folders = sorted(p.parent for p in CORPUS.glob("*/corpus.json"))
    if not folders:
        print("No papers read yet -- run: python scripts/build_corpus.py")
        return 1
    problems = [] if args.skip_checks else room_problems(*NEEDS[args.what, device])
    if problems:
        report_no_room(problems)
        return 1
    if device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("PyTorch cannot see the B580 (BUILD_PLAN.md, Day 1).")
        return 1

    print(f"building the {args.what} index on the {device.upper()} for {len(folders)} papers in "
          f"{CORPUS.relative_to(cfg.PROJECT_ROOT).as_posix()}/")
    try:
        build(args.what, folders, device, args.force)
    finally:
        unload_models()                       # hand the GPU memory back
    return 0


if __name__ == "__main__":
    sys.exit(main())
