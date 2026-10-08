# Read every paper of the dev and test splits into data/corpus/<paper_id>/, then check
# that the table or figure each question's answer lives in was found
# (BUILD_PLAN.md, Day 2, steps 2.2-2.3).
#
#   python scripts/build_corpus.py            # read the papers not read yet, then report
#   python scripts/build_corpus.py --force    # read every paper again
#
# Power-cut safe: a paper counts as done only when its corpus.json exists, and that file
# is written last, so an interrupted paper is simply read again on the next run.
#
# The report uses each question's LABEL ("Table 3") only, never its text: the test
# questions stay unseen. It writes results/day2_coverage.json for the thesis and, for
# every miss, an image of each page where the label appears, in data/debug/misses/.
import argparse
import json
import sys
import time
from collections import defaultdict

import pymupdf

from gea import config as cfg
from gea.ingest import draw_page, ingest_pdf
from gea.safeio import atomic_write_json, read_jsonl

SPLITS = cfg.PROJECT_ROOT / "eval" / "splits"
CORPUS = cfg.DATA_DIR / "corpus"


def read_papers(paper_ids: list, force: bool) -> None:
    for n, pid in enumerate(paper_ids, 1):
        out = CORPUS / pid
        if (out / "corpus.json").exists() and not force:
            continue
        out.mkdir(parents=True, exist_ok=True)
        start = time.time()
        p = ingest_pdf(cfg.DATA_DIR / "pdfs" / f"{pid}.pdf", out)
        print(f"  [{n}/{len(paper_ids)}] {pid}: {len(p['pages'])} pages, {len(p['chunks'])} chunks, "
              f"{len(p['tables'])} tables, {len(p['figures'])} figures  ({time.time() - start:.1f} s)")


def coverage(questions: list) -> tuple:
    # For each question: was a table/figure with its label (and kind) found in its paper?
    papers, found, misses = {}, 0, []
    for q in questions:
        if q["paper_id"] not in papers:
            papers[q["paper_id"]] = json.load(open(CORPUS / q["paper_id"] / "corpus.json", encoding="utf-8"))
        records = papers[q["paper_id"]][q["ref_kind"] + "s"]
        if any(r["label"] == q["ref_label"] for r in records):
            found += 1
        else:
            misses.append({"qid": q["qid"], "paper_id": q["paper_id"], "label": q["ref_label"]})
    return found, misses


def draw_misses(misses: list) -> int:
    # An image of every page where a missed label is printed, with what WAS found on it.
    out = cfg.DATA_DIR / "debug" / "misses"
    out.mkdir(parents=True, exist_ok=True)
    drawn = 0
    for m in {(m["paper_id"], m["label"]) for m in misses}:
        pid, label = m
        paper = json.load(open(CORPUS / pid / "corpus.json", encoding="utf-8"))
        doc = pymupdf.open(cfg.DATA_DIR / "pdfs" / f"{pid}.pdf")
        variants = {label, label.upper(), label.replace("Figure", "Fig.")}
        for page in doc:
            if any(page.search_for(v) for v in variants):
                image = draw_page(paper, CORPUS / pid, page.number + 1)
                image.save(out / f"{pid}_{label.replace(' ', '')}_p{page.number + 1}.png")
                drawn += 1
    return drawn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="read every paper again")
    args = ap.parse_args()

    splits = {name: read_jsonl(SPLITS / f"{name}.jsonl") for name in ("dev", "test")}
    paper_ids = list(dict.fromkeys(q["paper_id"] for qs in splits.values() for q in qs))
    print(f"reading {len(paper_ids)} papers into {CORPUS.relative_to(cfg.PROJECT_ROOT).as_posix()}/")
    read_papers(paper_ids, args.force)

    report = {}
    print("\ncoverage: is the answer's table/figure among the records?")
    for name, questions in splits.items():
        found, misses = coverage(questions)
        by_kind = defaultdict(lambda: [0, 0])
        for q in questions:
            by_kind[q["ref_kind"]][1] += 1
        for m in misses:
            by_kind[next(q["ref_kind"] for q in questions if q["qid"] == m["qid"])][0] += 1
        share = found / len(questions)
        kinds = ", ".join(f"{k}s {t - miss}/{t}" for k, (miss, t) in sorted(by_kind.items()))
        print(f"  {name:4s}: {found}/{len(questions)} = {share:.0%}   ({kinds})")
        for m in misses:
            print(f"        missed {m['label']:10s} in {m['paper_id']}  ({m['qid']})")
        report[name] = {"questions": len(questions), "found": found, "share": round(share, 4),
                        "misses": misses}

    totals = {"papers": len(paper_ids), "pages": 0, "chunks": 0, "tables": 0, "figures": 0,
              "found_by": defaultdict(int)}
    for pid in paper_ids:
        p = json.load(open(CORPUS / pid / "corpus.json", encoding="utf-8"))
        totals["pages"] += len(p["pages"])
        for key in ("chunks", "tables", "figures"):
            totals[key] += len(p[key])
        for r in p["tables"] + p["figures"]:
            totals["found_by"][r["found_by"]] += 1
    report["corpus"] = totals
    print(f"\ncorpus: {totals['papers']} papers, {totals['pages']} pages, {totals['chunks']} chunks, "
          f"{totals['tables']} tables, {totals['figures']} figures; regions found by {dict(totals['found_by'])}")

    atomic_write_json(cfg.PROJECT_ROOT / "results" / "day2_coverage.json", report)
    all_misses = report["dev"]["misses"] + report["test"]["misses"]
    if all_misses:
        print(f"\n{draw_misses(all_misses)} page image(s) of the misses in data/debug/misses/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
