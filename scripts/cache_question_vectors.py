# Day 7, before llama-server starts: cache ColQwen2's vectors for every question of a split.
# The region choice ("both": the region's words + ColQwen2's page score) uses the QUESTION's
# vectors, and ColQwen2 cannot load while llama-server holds the B580 -- so they are made
# first, in a GPU window, and the runner (scripts/run_eval.py) refuses to start without them.
#
#   python scripts/cache_question_vectors.py --split test
#
# Prints counts only, never the questions (the test set stays unseen). Safe to re-run: cached
# questions are skipped, and each vector file is written atomically (a power cut costs one).
import argparse
import hashlib
import sys
import time

from gea import config as cfg
from gea.device import report_no_room, room_problems
from gea.indexes import QUERY_CACHE, colqwen_device, page_query_vectors, unload_models
from gea.safeio import read_jsonl


def cached(question: str) -> bool:
    return (QUERY_CACHE / f"{hashlib.sha1(question.encode('utf-8')).hexdigest()}.pt").exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=("dev", "test"))
    args = ap.parse_args()
    questions = read_jsonl(cfg.PROJECT_ROOT / "eval" / "splits" / f"{args.split}.jsonl")
    todo = [q for q in questions if not cached(q["question"])]
    print(f"{args.split}: {len(questions) - len(todo)}/{len(questions)} question vectors already cached")
    if not todo:
        return 0
    device = colqwen_device()
    if device is None:
        report_no_room(room_problems(7, 4_500))
        return 1
    start = time.time()
    for i, q in enumerate(todo, 1):
        if page_query_vectors(q["question"]) is None:
            print("ColQwen2 could not run; stopped (the cached ones are kept)")
            return 1
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} done ({time.time() - start:.0f} s, {device})", flush=True)
    unload_models()
    print(f"{args.split}: {sum(cached(q['question']) for q in questions)}/{len(questions)} cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
