# Day 7, the second pass, step 2 -- with llama-server STOPPED: cache ColQwen2's vectors for
# every claim whose figure box still comes from the question's map (a photo: no printed
# words to box). Then `run_eval.py ... --rebox` replays those questions from the cache and
# N10b boxes each such claim with its own map.
#
#   python scripts/claim_vectors.py results/runs/full_test.jsonl [more run files ...]
#
# Prints counts only (no claim text: the test answers stay unread until Day 8).
import sys

from gea.device import report_no_room, room_problems
from gea.indexes import colqwen_device, page_query_vectors, query_cache_path, unload_models
from gea.safeio import read_jsonl


def claims_with_map_boxes(path) -> list:
    texts = []
    for r in read_jsonl(path):
        for b in r["boxes"]:
            if "(patch map)" in b["what"]:
                texts.append(r["claims"][b["claim"] - 1]["text"])
    return list(dict.fromkeys(texts))


def main(paths: list) -> int:
    texts = list(dict.fromkeys(t for p in paths for t in claims_with_map_boxes(p)))
    todo = [t for t in texts if not query_cache_path(t).exists()]
    print(f"{len(texts)} claim(s) with a map box; {len(texts) - len(todo)} already cached")
    if not todo:
        return 0
    device = colqwen_device()
    if device is None:
        report_no_room(room_problems(7, 4_500))
        return 1
    for t in todo:
        if page_query_vectors(t) is None:
            print("ColQwen2 could not run; stopped (the cached ones are kept)")
            return 1
    unload_models()
    print(f"cached {len(todo)} claim vector(s) on {device}; now: python scripts/run_eval.py ... --rebox")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/claim_vectors.py results/runs/<run>.jsonl [...]")
    sys.exit(main(sys.argv[1:]))
