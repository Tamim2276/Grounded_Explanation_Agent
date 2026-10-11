# How often does N1 give a sub-goal the kind of evidence the answer is in (BUILD_PLAN.md
# Day 5)? On the 20 dev questions, for the current prompt (PROMPT_VERSION). SPIQA
# references are tables or figures, so the reference kind is what N1 should find; "text"
# can still be a fair choice for some questions, so this is a rough measure.
#
#   python scripts/eval_n1.py        # needs llama-server, or replies cached for this prompt
#
# Adds the result for this prompt version to results/day5_n1.json (earlier versions kept).
import json
import sys
from collections import Counter

import gea.llm as llm
from gea import config as cfg
from gea.question import llm_sub_goals, rule_sub_goals
from gea.safeio import atomic_write_json, read_jsonl

OUT = cfg.PROJECT_ROOT / "results" / "day5_n1.json"


def main() -> int:
    questions = read_jsonl(cfg.PROJECT_ROOT / "eval" / "splits" / "dev.jsonl")
    rows = []
    for q in questions:
        model = [g["modality"] for g in llm_sub_goals(q["question"])]
        rules = [g["modality"] for g in rule_sub_goals(q["question"])]
        rows.append({"qid": q["qid"], "ref_kind": q["ref_kind"], "model": model, "rules": rules,
                     "model_right": q["ref_kind"] in model, "rules_right": q["ref_kind"] in rules})
    n = len(rows)
    result = {"questions": n,
              "model_right": sum(r["model_right"] for r in rows),
              "rules_right": sum(r["rules_right"] for r in rows),
              "model_kinds": dict(Counter(k for r in rows for k in r["model"])),
              "by_kind": {kind: f"{sum(r['model_right'] for r in rows if r['ref_kind'] == kind)}/"
                                f"{sum(r['ref_kind'] == kind for r in rows)}" for kind in ("table", "figure")},
              "rows": rows}
    saved = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    saved[llm.PROMPT_VERSION] = result
    atomic_write_json(OUT, saved)
    print(f"prompt {llm.PROMPT_VERSION}: N1 gave the reference's kind in {result['model_right']}/{n} "
          f"(tables {result['by_kind']['table']}, figures {result['by_kind']['figure']}); "
          f"kinds chosen {result['model_kinds']}; the regex stand-in: {result['rules_right']}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
