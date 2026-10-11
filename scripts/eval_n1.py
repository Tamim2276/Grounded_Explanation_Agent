# How often does N1 give a sub-goal the kind of evidence the answer is in (BUILD_PLAN.md
# Day 5)? On the 20 dev questions, for the current prompt (PROMPT_VERSION). SPIQA
# references are tables or figures, so the reference kind is what N1 should find; "text"
# can still be a fair choice for some questions, so this is a rough measure.
#
#   python scripts/eval_n1.py        # the model part needs llama-server or cached replies
#
# Adds to results/day5_n1.json (earlier entries kept): the model's choice for this prompt
# version (when the model can answer), and "vote": the kind voted by the cfg.KIND_VOTE_K most
# similar labelled SPIQA questions from papers outside dev and test (gea/kinds.py).
import json
import sys
from collections import Counter

import gea.llm as llm
from gea import config as cfg
from gea.kinds import kind_of
from gea.question import llm_sub_goals, rule_sub_goals
from gea.safeio import atomic_write_json, read_jsonl

OUT = cfg.PROJECT_ROOT / "results" / "day5_n1.json"


def main() -> int:
    questions = read_jsonl(cfg.PROJECT_ROOT / "eval" / "splits" / "dev.jsonl")
    saved = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    votes = [(q, *kind_of(q["question"])) for q in questions]
    vote_rows = [{"qid": q["qid"], "ref_kind": q["ref_kind"], "vote": kind, "votes": f"{v}/{k}",
                  "right": kind == q["ref_kind"]} for q, kind, v, k in votes]
    saved["vote"] = {"k": cfg.KIND_VOTE_K, "questions": len(vote_rows), "right": sum(r["right"] for r in vote_rows),
                     "by_kind": {kind: f"{sum(r['right'] for r in vote_rows if r['ref_kind'] == kind)}/"
                                       f"{sum(r['ref_kind'] == kind for r in vote_rows)}" for kind in ("table", "figure")},
                     "rows": vote_rows}
    v = saved["vote"]
    print(f"vote of the {v['k']} nearest labelled questions: {v['right']}/{v['questions']} "
          f"(tables {v['by_kind']['table']}, figures {v['by_kind']['figure']})")
    atomic_write_json(OUT, saved)

    rows = []
    for q in questions:
        try:
            model = [g["modality"] for g in llm_sub_goals(q["question"])]
        except (OSError, ValueError, KeyError):
            print(f"model: no server and no cached reply for prompt {llm.PROMPT_VERSION} -- model part skipped")
            return 0
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
    saved[llm.PROMPT_VERSION] = result
    atomic_write_json(OUT, saved)
    print(f"prompt {llm.PROMPT_VERSION}: N1 gave the reference's kind in {result['model_right']}/{n} "
          f"(tables {result['by_kind']['table']}, figures {result['by_kind']['figure']}); "
          f"kinds chosen {result['model_kinds']}; the regex stand-in: {result['rules_right']}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
