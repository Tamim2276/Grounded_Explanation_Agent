# Day 8.6: every score x every system x {all, multi-hop}, with 95% bootstrap intervals, and the
# paired differences between the full system and each other run -> results/summary.json.
#
#   python scripts/score.py
#
# Reads results/runs/ and results/judged/ (the judge), eval/multihop.json (8.5) and, once they
# exist, the hand work: eval/annotations.csv (boxes, for CGS), eval/judge_check.csv (the judge
# check) and eval/attribution.csv (MAP). A part whose input is missing says "waiting for ...".
import csv
import json
from statistics import median

from gea import config as cfg
from gea.metrics import (aes, bootstrap_diff, bootstrap_mean, cgs, cohen_kappa, rf, search_calls,
                         visual_labels)
from gea.safeio import atomic_write_json, read_jsonl

ROOT = cfg.PROJECT_ROOT
EVAL = ROOT / "eval"
RUNS = {"full": "full_test", "single_pass": "single_pass_test", "text_only": "text_only_test",
        "full_strict": "full_strict_test", "full_gemini": "full_gemini_test"}
# Full vs the two baselines (RQ1, RQ2), the strict ablation, and the stronger model.
PAIRS = [("full", "single_pass"), ("full", "text_only"), ("full", "full_strict"), ("full_gemini", "full")]


def interval(values) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "ci95": None, "n": 0}
    mean, ci = bootstrap_mean(values)
    return {"mean": round(mean, 4), "ci95": [round(v, 4) for v in ci], "n": len(values)}


def gold_boxes() -> dict:
    # qid -> [(page, bbox)] from the hand-drawn boxes; a "cannot box" row leaves the qid out.
    path = EVAL / "annotations.csv"
    if not path.exists():
        return {}
    gold = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["x0"]:
                gold.setdefault(row["qid"], []).append(
                    (int(row["page"]), tuple(float(row[k]) for k in ("x0", "y0", "x1", "y1"))))
    return gold


def per_question(name: str, multihop: set, gold: dict) -> list:
    runs = {r["qid"]: r for r in read_jsonl(ROOT / "results" / "runs" / f"{RUNS[name]}.jsonl")}
    judged = {j["qid"]: j for j in read_jsonl(ROOT / "results" / "judged" / f"{RUNS[name]}.jsonl")}
    rows = []
    for qid, r in runs.items():
        j = judged.get(qid)
        out = j["outcome"] if j else None
        labels = visual_labels(r)
        rows.append({
            "qid": qid, "multihop": qid in multihop, "outcome": out, "best_effort": bool(r.get("best_effort")),
            "correct": None if out is None else float(out == "correct"),
            "wrong": None if out is None else float(out == "wrong"),
            "abstained": None if out is None else float(out == "abstained"),
            "hit": float(bool(r.get("reference_in_buffer"))),
            "rf": rf(labels, r["reference"]),
            "aes": aes(2 if qid in multihop else 1, search_calls(r)),
            "search_calls": search_calls(r), "model_calls": r["model_calls"], "seconds": r["seconds"],
            "cgs": cgs(r["boxes"], gold[qid]) if qid in gold else None,
            "answered": out in ("correct", "wrong"),
        })
    return rows


def summarise(rows: list) -> dict:
    judged = [r for r in rows if r["outcome"] is not None]
    answered = [r for r in judged if r["answered"]]
    split = {}
    for kind, keep in (("confident", False), ("flagged", True)):
        part = [r for r in answered if r["best_effort"] == keep]
        split[kind] = {"answered": len(part), "correct": sum(r["outcome"] == "correct" for r in part),
                       "wrong": sum(r["outcome"] == "wrong" for r in part)}
    boxed = [r for r in rows if r["cgs"] is not None]
    return {
        "questions": len(rows), "judged": len(judged),
        "counts": {k: sum(r["outcome"] == k for r in judged) for k in ("correct", "wrong", "abstained", "judge_error")},
        "correct": interval(r["correct"] for r in judged),
        "wrong": interval(r["wrong"] for r in judged),
        "abstained": interval(r["abstained"] for r in judged),
        "precision_when_answering": interval(r["correct"] for r in answered),   # correct / answered
        "confident_vs_flagged": split,
        "hit_rate": interval(r["hit"] for r in rows),
        "rf": interval(r["rf"] for r in rows),                                  # None: nothing visual fetched
        "aes": interval(r["aes"] for r in rows),
        "search_calls_per_question": round(sum(r["search_calls"] for r in rows) / len(rows), 2),
        "model_calls_per_question": round(sum(r["model_calls"] for r in rows) / len(rows), 2),
        "median_seconds": round(median(r["seconds"] for r in rows), 1),
        "cgs_all": interval(r["cgs"] for r in boxed) if boxed else "waiting for eval/annotations.csv",
        "cgs_answered": interval(r["cgs"] for r in boxed if r["answered"]) if boxed else None,
        "cgs_at_least_0.5": round(sum(r["cgs"] >= 0.5 for r in boxed) / len(boxed), 4) if boxed else None,
    }


def paired(a: list, b: list, key: str) -> dict:
    # Full minus the other run, on the questions both have a value for, in the same order.
    other = {r["qid"]: r[key] for r in b}
    pairs = [(r[key], other[r["qid"]]) for r in a if r[key] is not None and other.get(r["qid"]) is not None]
    if not pairs:
        return None
    diff, ci = bootstrap_diff([p[0] for p in pairs], [p[1] for p in pairs])
    return {"diff": round(diff, 4), "ci95": [round(v, 4) for v in ci], "n": len(pairs),
            "beyond_luck": ci[0] > 0 or ci[1] < 0}


def judge_check() -> dict:
    path = EVAL / "judge_check.csv"
    if not path.exists():
        return "waiting for eval/judge_check.csv"
    chosen = {g["item"]: g for g in json.loads((EVAL / "annotation_set.json").read_text(encoding="utf-8"))["grade"]}
    human, model, unsure = [], [], 0
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            g = chosen[int(row["item"])]
            j = next(x for x in read_jsonl(ROOT / "results" / "judged" / f"{g['run']}_test.jsonl") if x["qid"] == g["qid"])
            if row["verdict"] == "unsure":
                unsure += 1
                continue
            human.append(row["verdict"]); model.append(j["outcome"])
    agree = sum(h == m for h, m in zip(human, model))
    return {"graded": len(human), "unsure": unsure, "agreement": round(agree / len(human), 4),
            "cohen_kappa": round(cohen_kappa(human, model), 4),
            "human_correct": human.count("correct"), "judge_correct": model.count("correct")}


def attribution() -> dict:
    # MAP: the hand marks of eval/attribution.csv (qid, run, claim, right_source: yes/no).
    path = EVAL / "attribution.csv"
    if not path.exists():
        return "waiting for eval/attribution.csv"
    marks = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            marks.setdefault(row["run"], []).append(float(row["right_source"] == "yes"))
    return {run: interval(v) for run, v in marks.items()}


def main():
    multihop = set(json.loads((EVAL / "multihop.json").read_text(encoding="utf-8"))["qids"])
    gold = gold_boxes()
    rows = {name: per_question(name, multihop, gold) for name in RUNS}
    summary = {"systems": {}, "comparisons": {}, "judge_check": judge_check(), "map": attribution(),
               "multihop_questions": len(multihop), "annotated_boxes": len(gold)}
    for name, r in rows.items():
        summary["systems"][name] = {"all": summarise(r), "multihop": summarise([x for x in r if x["multihop"]])}
    for a, b in PAIRS:
        for subset, keep in (("all", lambda x: True), ("multihop", lambda x: x["multihop"])):
            ra, rb = [x for x in rows[a] if keep(x)], [x for x in rows[b] if keep(x)]
            summary["comparisons"][f"{a} - {b} ({subset})"] = {
                key: paired(ra, rb, key) for key in ("correct", "wrong", "abstained", "hit", "rf", "aes", "cgs")}
    atomic_write_json(ROOT / "results" / "summary.json", summary)

    print(f"{'system':<12} {'correct':>8} {'wrong':>7} {'abstain':>8} {'hit':>6} {'RF':>6} {'AES':>6} {'CGS':>6}")
    for name in RUNS:
        s = summary["systems"][name]["all"]
        m = lambda k: f"{s[k]['mean']:.2f}" if isinstance(s[k], dict) and s[k]["mean"] is not None else "  -"
        print(f"{name:<12} {m('correct'):>8} {m('wrong'):>7} {m('abstained'):>8} {m('hit_rate'):>6} {m('rf'):>6} "
              f"{m('aes'):>6} {m('cgs_all'):>6}   (judged {s['judged']}/{s['questions']})")
    print("-> results/summary.json")


if __name__ == "__main__":
    main()
