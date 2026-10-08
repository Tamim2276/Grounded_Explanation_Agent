# Choose the practice (dev) and exam (test) questions from SPIQA test-A
# (BUILD_PLAN.md, Day 2, step 2.1).
#
#   python scripts/select_questions.py
#
# Writes eval/splits/dev.jsonl, eval/splits/test.jsonl and eval/splits/summary.json.
# The same seed always gives the same split, so anyone can rebuild exactly these
# questions. The files are in git: they ARE the experiment.
#
# Rules:
#   * Papers are shuffled with a fixed seed, then dealt out whole: a paper's
#     questions are never split between dev and test, or you would practise on the
#     exam's paper.
#   * At most MAX_PER_PAPER questions per paper, so a few question-heavy papers
#     cannot dominate the test set; at most DEV_MAX_PER_PAPER in dev, so tuning sees
#     several different paper layouts, not two or three.
#   * Dev first (DEV_SIZE questions), then test (TEST_SIZE questions).
#   * Test question texts are never printed: you must not see the exam before Day 7.
import json
import random
import re
import sys
from collections import Counter

from gea import config as cfg
from gea.safeio import atomic_write_json, atomic_write_text

SEED = 42
DEV_SIZE = 20
TEST_SIZE = 150
MAX_PER_PAPER = 8        # test: at most 8 questions from one paper
DEV_MAX_PER_PAPER = 4    # dev: at most 4, so the 20 practice questions cover ~5 different papers

SPIQA_JSON = cfg.DATA_DIR / "spiqa" / "test-A" / "SPIQA_testA.json"
PDF_DIR = cfg.DATA_DIR / "pdfs"
OUT_DIR = cfg.PROJECT_ROOT / "eval" / "splits"

# "1611.04684v1-Table1-1.png" -> Table 1;  "1803.02750v3-TableII-1.png" -> Table II
# (some papers number their tables with Roman numerals, IEEE style)
REFERENCE = re.compile(r"-(Figure|Table)(\d+|[IVXLC]+)-\d+\.png$")


def label_of(reference: str):
    # The label of the table or figure a question points at, or None.
    m = REFERENCE.search(reference)
    return f"{m.group(1)} {m.group(2)}" if m else None


def questions_of(paper_id: str, paper: dict) -> list:
    # Every usable question of one paper, as one flat record each, in SPIQA's order.
    records = []
    for i, qa in enumerate(paper["qa"]):
        label = label_of(qa["reference"])
        info = paper["all_figures"].get(qa["reference"])
        if label is None or info is None:
            print(f"  skipped {paper_id} question {i}: unusable reference {qa['reference']!r}")
            continue
        records.append({
            "qid": f"{paper_id}#{i}",            # stable: the question's place in SPIQA
            "paper_id": paper_id,
            "question": qa["question"],
            "answer": qa["answer"],
            "explanation": qa["explanation"],
            "reference": qa["reference"],
            "ref_label": label,                    # "Table 1", "Figure 3", "Table II"
            "ref_kind": info["content_type"],      # "table" | "figure"
            "ref_caption": info["caption"],
        })
    return records


def deal(papers: list, size: int, per_paper: int) -> tuple:
    # Take papers from the front of the list until `size` questions are collected,
    # at most `per_paper` from each. Returns (questions, papers used).
    taken, used = [], []
    while len(taken) < size and papers:
        paper_id, records = papers.pop(0)
        taken += records[: min(per_paper, size - len(taken))]
        used.append(paper_id)
    return taken, used


def describe(name: str, questions: list, papers: list) -> dict:
    kinds = Counter(q["ref_kind"] for q in questions)
    roman = sum(bool(re.search(r" [IVXLC]+$", q["ref_label"])) for q in questions)
    print(f"  {name:4s}: {len(questions):3d} questions from {len(papers):2d} papers  "
          f"({kinds['table']} table, {kinds['figure']} figure; {roman} with Roman-numeral labels)")
    return {"questions": len(questions), "papers": len(papers), "paper_ids": papers,
            "table": kinds["table"], "figure": kinds["figure"], "roman_labels": roman}


def main() -> int:
    spiqa = json.load(open(SPIQA_JSON, encoding="utf-8"))
    have_pdf = {p.stem for p in PDF_DIR.glob("*.pdf")}
    print(f"SPIQA test-A: {len(spiqa)} papers, {sum(len(p['qa']) for p in spiqa.values())} questions; "
          f"{len(have_pdf)} PDFs on disk")

    # Only papers whose PDF we have; then a fixed shuffle.
    paper_ids = sorted(pid for pid in spiqa if pid.replace("/", "_") in have_pdf)
    random.Random(SEED).shuffle(paper_ids)
    papers = [(pid, questions_of(pid, spiqa[pid])) for pid in paper_ids]
    papers = [(pid, recs) for pid, recs in papers if recs]

    dev, dev_papers = deal(papers, DEV_SIZE, DEV_MAX_PER_PAPER)
    test, test_papers = deal(papers, TEST_SIZE, MAX_PER_PAPER)
    assert len(dev) == DEV_SIZE and len(test) == TEST_SIZE, "not enough questions"
    assert not set(dev_papers) & set(test_papers), "a paper landed in both splits"

    print("\nsplits:")
    summary = {"seed": SEED, "max_per_paper": {"dev": DEV_MAX_PER_PAPER, "test": MAX_PER_PAPER},
               "source": "SPIQA test-A",
               "dev": describe("dev", dev, dev_papers),
               "test": describe("test", test, test_papers)}

    for name, questions in (("dev", dev), ("test", test)):
        atomic_write_text(OUT_DIR / f"{name}.jsonl",
                          "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions))
    atomic_write_json(OUT_DIR / "summary.json", summary)
    print(f"\nwrote {OUT_DIR.relative_to(cfg.PROJECT_ROOT).as_posix()}/dev.jsonl, test.jsonl, summary.json")

    print("\ntwo practice questions (test questions are never shown):")
    for q in dev[:2]:
        print(f"  [{q['qid']}] {q['question']}\n      -> answer in {q['ref_label']} ({q['ref_kind']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
