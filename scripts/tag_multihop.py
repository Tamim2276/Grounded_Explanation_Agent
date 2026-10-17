# Day 8.5: the multi-hop subset -- the test questions that need the narrative text AND a table or
# figure. Every SPIQA question points at a table or figure; it also needs the text when it asks
# for a reason or an implication of what the table or figure shows ("why", "explain why", "what
# does this suggest", "a possible explanation"): the result is in the table or figure, the reason
# in the text. A fixed rule over the question text alone, so the subset was chosen without
# looking at any system's output and anyone can rebuild it.
#
#   python scripts/tag_multihop.py       # -> eval/multihop.json
import re

from gea import config as cfg
from gea.safeio import atomic_write_json, read_jsonl

RULE = re.compile(r"\bwhy\b|\bsuggests?\b|\bpossible explanation\b", re.IGNORECASE)   # not "explanation method"


def main():
    questions = read_jsonl(cfg.PROJECT_ROOT / "eval" / "splits" / "test.jsonl")
    tagged = [q["qid"] for q in questions if RULE.search(q["question"])]
    atomic_write_json(cfg.PROJECT_ROOT / "eval" / "multihop.json",
                      {"rule": RULE.pattern, "needed_hops": 2, "qids": tagged})
    kinds = [q["ref_kind"] for q in questions if q["qid"] in tagged]
    print(f"{len(tagged)} of {len(questions)} test questions are multi-hop "
          f"({kinds.count('table')} table, {kinds.count('figure')} figure) -> eval/multihop.json")


if __name__ == "__main__":
    main()
