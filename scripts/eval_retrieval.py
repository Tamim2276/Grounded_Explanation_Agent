# First retrieval numbers for RQ2 (BUILD_PLAN.md, Day 3, step 3.6), on the 20 dev
# questions. Needs both indexes (scripts/build_indexes.py text, then pages).
#
#   python scripts/eval_retrieval.py          # on the B580 when it is free (~1 min)
#   python scripts/eval_retrieval.py --cpu    # on the CPU (needs ~7 GB of free memory)
#
# For every question, the "reference" is the table or figure its answer lives in.
#   page hit@k: its page is among the k pages ColQwen2 scores highest (MaxSim)
#   text hit@3: one of the 3 best text-index rows is the caption, names the reference
#               ("Table 2", "Fig. 2"), or is on its page; "strict" leaves out "on its page"
#   random:     what picking k pages blindly would score, for comparison
# Writes results/day3_retrieval.json. Dev only: the test questions stay unseen.
import argparse
import json
import sys
import time

import torch

from gea import config as cfg
from gea.device import report_no_room, room_problems
from gea.indexes import (embed_page_query, embed_text, load_page_index, load_text_index, maxsim,
                         unload_models)
from gea.safeio import atomic_write_json, read_jsonl
from gea.text import names_label

CORPUS = cfg.DATA_DIR / "corpus"
OUT = cfg.PROJECT_ROOT / "results" / "day3_retrieval.json"


def text_match(top: list, ref: dict):
    # Why the text index counts as having found the reference, strongest reason first.
    # "same page" alone can be a coincidence, so the summary reports a strict number too.
    if any(r["id"] == ref["id"] for r in top):
        return "caption"
    if any(names_label(r.get("text", ""), ref["label"]) for r in top):
        return "names label"
    if any(r["page"] == ref["page"] for r in top):
        return "same page"
    return None


def evaluate(q: dict, paper: dict, pages: list, text: dict, device: str) -> dict:
    ref = next(r for r in paper[q["ref_kind"] + "s"] if r["label"] == q["ref_label"])

    scores = maxsim(embed_page_query(q["question"], device), [p["vectors"] for p in pages])
    order = sorted(range(len(pages)), key=lambda i: -scores[i])          # best page first
    ranked = [pages[i]["page"] for i in order]

    records = {r["id"]: r for r in paper["chunks"] + paper["tables"] + paper["figures"]}
    _, rows = text["faiss"].search(embed_text([q["question"]], "query", device), 3)
    top3 = [records[text["ids"][r]] for r in rows[0]]
    match = text_match(top3, ref)

    return {"qid": q["qid"], "ref_label": ref["label"], "ref_page": ref["page"], "pages": len(pages),
            "page_rank": ranked.index(ref["page"]) + 1, "top3_pages": ranked[:3],
            "top3_scores": [round(scores[i], 2) for i in order[:3]],
            "top3_text": [r["id"] for r in top3], "text_match": match, "text_hit3": match is not None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu", action="store_true", help="run ColQwen2 on the CPU instead of the B580")
    ap.add_argument("--skip-checks", action="store_true", help="start even if memory looks short")
    args = ap.parse_args()
    device = "cpu" if args.cpu else "xpu"

    problems = [] if args.skip_checks else room_problems(*((7, None) if args.cpu else (7, 4_500)))
    if problems:
        report_no_room(problems)
        return 1
    if device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("PyTorch cannot see the B580 (BUILD_PLAN.md, Day 1). Use --cpu.")
        return 1

    questions = read_jsonl(cfg.PROJECT_ROOT / "eval" / "splits" / "dev.jsonl")
    start, results, loaded = time.time(), [], {}
    try:
        for q in questions:
            pid = q["paper_id"]
            if pid not in loaded:
                folder = CORPUS / pid
                loaded[pid] = (json.loads((folder / "corpus.json").read_text(encoding="utf-8")),
                               load_page_index(folder), load_text_index(folder))
            r = evaluate(q, *loaded[pid], device)
            results.append(r)
            print(f"  {r['qid']:18s} {r['ref_label']:10s} on page {r['ref_page']:2d}/{r['pages']:2d}: "
                  f"page rank {r['page_rank']:2d}  (top 3: {r['top3_pages']})  text hit@3: "
                  f"{'yes' if r['text_hit3'] else 'no'}", flush=True)
    finally:
        unload_models()

    n = len(results)
    summary = {"questions": n,
               "page_hit@1": sum(r["page_rank"] == 1 for r in results) / n,
               "page_hit@3": sum(r["page_rank"] <= 3 for r in results) / n,
               "text_hit@3": sum(r["text_hit3"] for r in results) / n,
               "text_hit@3_strict": sum(r["text_match"] in ("caption", "names label") for r in results) / n,
               "random_page_hit@1": sum(1 / r["pages"] for r in results) / n,
               "random_page_hit@3": sum(min(3, r["pages"]) / r["pages"] for r in results) / n}
    summary = {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}
    atomic_write_json(OUT, {"split": "dev", "device": device, "page_model": "vidore/colqwen2-v1.0",
                            "text_model": "allenai/specter2 (+ adhoc_query for questions)",
                            "summary": summary, "questions": results})
    print(f"\n{n} dev questions in {time.time() - start:.0f} s")
    print(f"  page hit@1 {summary['page_hit@1']:.0%}   (random {summary['random_page_hit@1']:.0%})")
    print(f"  page hit@3 {summary['page_hit@3']:.0%}   (random {summary['random_page_hit@3']:.0%})")
    print(f"  text hit@3 {summary['text_hit@3']:.0%}   "
          f"(strict, caption or label named: {summary['text_hit@3_strict']:.0%})")
    print(f"saved {OUT.relative_to(cfg.PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
