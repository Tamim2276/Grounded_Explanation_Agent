# Which way should N6a rank a real paper's paragraphs (BUILD_PLAN.md, Day 3, step 3.5)?
# Runs the real text retriever on the 20 dev questions three ways -- keyword, SPECTER2,
# hybrid (both, by reciprocal rank fusion) -- on the CPU, and writes
# results/day3_text_search.json.
#
#   python scripts/compare_text_search.py
#
# N6a returns paragraphs only, so a hit means one of them is about the question's
# reference table or figure:
#   strict:  the paragraph names it ("Table 2", "Fig. 2")
#   lenient: ... or sits on the same page (which can be a coincidence)
# Chosen on dev; the test questions stay unseen.
import sys
import time

from gea import config as cfg
from gea.corpus import get_corpus
from gea.device import report_no_room, room_problems
from gea.retrieval import text_retriever
from gea.safeio import atomic_write_json, read_jsonl
from gea.text import names_label

METHODS = ("keyword", "specter2", "hybrid")
OUT = cfg.PROJECT_ROOT / "results" / "day3_text_search.json"


def judge(found: list, ref: dict) -> dict:
    named = [names_label(e.content, ref["label"]) for e in found]
    return {"top3": [e.source for e in found],
            "strict@1": bool(named[:1] and named[0]),
            "strict@3": any(named),
            "lenient@3": any(named) or any(e.page == ref["page"] for e in found)}


def main() -> int:
    problems = room_problems(need_gb=2.5)
    if problems:
        report_no_room(problems)
        return 1
    questions = read_jsonl(cfg.PROJECT_ROOT / "eval" / "splits" / "dev.jsonl")
    start, rows = time.time(), []
    with cfg.override(BACKEND="real", TRACE=False):
        for q in questions:
            paper = get_corpus(q["paper_id"])
            ref = next(r for r in paper[q["ref_kind"] + "s"] if r["label"] == q["ref_label"])
            row = {"qid": q["qid"], "ref_label": ref["label"], "ref_page": ref["page"]}
            for how in METHODS:
                with cfg.override(TEXT_SEARCH=how):
                    row[how] = judge(text_retriever(paper, q["question"], k=3), ref)
            rows.append(row)

    n = len(rows)
    summary = {how: {m: round(sum(r[how][m] for r in rows) / n, 4)
                     for m in ("strict@1", "strict@3", "lenient@3")} for how in METHODS}
    atomic_write_json(OUT, {"split": "dev", "questions": n, "k": 3, "rrf_k": cfg.RRF_K,
                            "summary": summary, "per_question": rows})
    print(f"{n} dev questions, real N6a, top 3 paragraphs ({time.time() - start:.0f} s)\n")
    print(f"  {'method':9s} {'strict@1':>9s} {'strict@3':>9s} {'lenient@3':>10s}")
    for how in METHODS:
        s = summary[how]
        print(f"  {how:9s} {s['strict@1']:9.0%} {s['strict@3']:9.0%} {s['lenient@3']:10.0%}")
    print(f"saved {OUT.relative_to(cfg.PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
