# Day 8.2 and 8.3: prepare the hand work -- the 50 questions to box and the 30 answers to grade.
#
#   python scripts/prepare_annotation.py
#
# The 50: random.Random(42), 25 table and 25 figure references from the test split, chosen from
# the split alone, BEFORE any system output is looked at (a choice made after looking could
# favour questions where the boxes look good). The 30: random.Random(42), 6 published answers
# from each of the five test runs; the grader sees neither the system nor the judge's verdict.
#
# Writes eval/annotation_set.json (the choice, in git: it is part of the experiment) and
# data/annotate/ (the pages at 144 dpi and annotate.html, which works offline: copy the folder
# to a laptop). The marks come back as eval/annotations.csv and eval/judge_check.csv.
import json
import random
import shutil

import pymupdf

from gea import config as cfg
from gea.metrics import answer_text, published
from gea.safeio import atomic_write_json, atomic_write_text, read_jsonl

SEED = 42
PER_KIND = 25            # 25 tables + 25 figures
GRADE = 30               # answers graded by hand to check the judge
DPI = 144                # 2 pixels per PDF point: Paint's pixel position / 2 = points
RUNS = ("full", "full_strict", "single_pass", "text_only", "full_gemini")

SPLIT = cfg.PROJECT_ROOT / "eval" / "splits" / "test.jsonl"
SET_FILE = cfg.PROJECT_ROOT / "eval" / "annotation_set.json"
OUT = cfg.DATA_DIR / "annotate"
TOOL = cfg.PROJECT_ROOT / "scripts" / "annotate_tool.html"


def choose_boxes(questions: list) -> list:
    rng = random.Random(SEED)
    chosen = []
    for kind in ("table", "figure"):
        chosen += rng.sample([q for q in questions if q["ref_kind"] == kind], PER_KIND)
    order = {q["qid"]: i for i, q in enumerate(questions)}
    return sorted(chosen, key=lambda q: order[q["qid"]])      # paper by paper, as in the split


def choose_answers() -> list:
    # GRADE / 5 published answers from each run (a plain draw from the pool gave Gemini 3 of 30),
    # then shuffled, so the order does not tell the grader which system wrote an answer.
    rng, picked = random.Random(SEED), []
    for run in RUNS:
        pool = [r for r in read_jsonl(cfg.PROJECT_ROOT / "results" / "runs" / f"{run}_test.jsonl") if published(r)]
        picked += [(run, r) for r in rng.sample(pool, GRADE // len(RUNS))]
    rng.shuffle(picked)
    return [{"item": i + 1, "run": run, "qid": r["qid"], "question": r["question"], "gold": r["gold"],
             "answer": answer_text(r)} for i, (run, r) in enumerate(picked)]


def main():
    questions = read_jsonl(SPLIT)
    boxes, answers = choose_boxes(questions), choose_answers()
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    items = []
    for q in boxes:
        corpus = json.loads((cfg.DATA_DIR / "corpus" / q["paper_id"] / "corpus.json").read_text(encoding="utf-8"))
        region = next(r for r in corpus["tables"] + corpus["figures"] if r["label"] == q["ref_label"])
        png = f"pages/{q['qid'].replace('#', '_')}.png"
        with pymupdf.open(corpus["pdf"]) as doc:
            page = doc[region["page"] - 1]
            page.get_pixmap(dpi=DPI).save(OUT / png)
            size = [round(page.rect.width, 2), round(page.rect.height, 2)]
        items.append({"qid": q["qid"], "kind": q["ref_kind"], "label": q["ref_label"], "question": q["question"],
                      "gold": q["answer"], "page": region["page"], "page_size": size,
                      "region": [round(v, 1) for v in region["bbox"]], "png": png})
    atomic_write_json(SET_FILE, {"seed": SEED, "boxes": [i["qid"] for i in items],
                                 "grade": [{k: a[k] for k in ("item", "run", "qid")} for a in answers]})
    # The tool reads its data from a script tag: a page opened from disk may not fetch JSON.
    blind = [{k: a[k] for k in ("item", "question", "gold", "answer")} for a in answers]
    atomic_write_text(OUT / "items.js", "const BOXES = " + json.dumps(items, ensure_ascii=False, indent=1)
                      + ";\nconst ANSWERS = " + json.dumps(blind, ensure_ascii=False, indent=1) + ";\n")
    shutil.copyfile(TOOL, OUT / "annotate.html")
    tables = sum(i["kind"] == "table" for i in items)
    print(f"{len(items)} questions to box ({tables} tables, {len(items) - tables} figures), "
          f"{len(answers)} answers to grade -> {OUT}")
    print(f"the choice: {SET_FILE.relative_to(cfg.PROJECT_ROOT)}; open {OUT / 'annotate.html'} in a browser")


if __name__ == "__main__":
    main()
