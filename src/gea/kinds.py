# Which kind of evidence a sub-question needs (table or figure), learned from labelled
# examples instead of asked of the language model (BUILD_PLAN.md Day 5).
#
# SPIQA labels every question with the table or figure its answer is in. The questions of
# the SPIQA test-A papers that are in NEITHER the dev NOR the test split (469 questions
# from 82 papers) are the examples: a sub-question takes the majority kind of its
# cfg.KIND_VOTE_K most similar examples (SPECTER2 question vectors, cosine). Because the
# example papers are never evaluated, nothing about the dev or test questions leaks in.
#
# On the dev questions this gives the reference's kind 17/20 times (the 7B model 8/20,
# the regex stand-in 2/20). It can only say table or figure -- SPIQA labels nothing as
# text -- which suits questions about a paper's tables and figures.
import json
from functools import lru_cache

import numpy as np

from gea import config as cfg
from gea.safeio import atomic_path, read_jsonl

SPIQA_JSON = cfg.DATA_DIR / "spiqa" / "test-A" / "SPIQA_testA.json"
EXAMPLES = cfg.DATA_DIR / "cache" / "kind_examples.npz"


def evaluated_papers() -> list:
    # Every paper of the dev and test splits: never used as an example.
    return sorted({q["paper_id"] for split in ("dev", "test")
                   for q in read_jsonl(cfg.PROJECT_ROOT / "eval" / "splits" / f"{split}.jsonl")})


def example_questions(exclude: list) -> tuple:
    # (questions, kinds) of the SPIQA test-A papers not in `exclude`.
    spiqa = json.loads(SPIQA_JSON.read_text(encoding="utf-8"))
    pairs = [(qa["question"], "table" if "Table" in qa["reference"] else "figure")
             for pid, paper in sorted(spiqa.items()) if pid not in set(exclude) for qa in paper["qa"]]
    return [q for q, _ in pairs], [k for _, k in pairs]


def build_examples() -> int:
    # Embed the example questions once (SPECTER2, question adapter, CPU, ~10 s) and save
    # them with the list of excluded papers, so a change of the splits is noticed.
    from gea.indexes import embed_text
    exclude = evaluated_papers()
    questions, kinds = example_questions(exclude)
    with atomic_path(EXAMPLES) as tmp:
        np.savez(tmp, vectors=embed_text(questions, kind="query"), kinds=np.array(kinds),
                 questions=np.array(questions), excluded=np.array(exclude))
    return len(questions)


@lru_cache(maxsize=1)
def examples() -> tuple:
    # (vectors, kinds), rebuilt when missing or when the dev/test papers changed.
    if not EXAMPLES.exists() or list(np.load(EXAMPLES)["excluded"]) != evaluated_papers():
        build_examples()
    data = np.load(EXAMPLES)
    return data["vectors"], data["kinds"]


def vote(vector: np.ndarray, vectors: np.ndarray, kinds: np.ndarray, k: int) -> tuple:
    # The majority kind among the k most similar examples: (kind, votes for it).
    nearest = np.argsort(-(vectors @ vector), kind="stable")[:k]
    tables = int((kinds[nearest] == "table").sum())
    return ("table", tables) if tables * 2 > k else ("figure", k - tables)


def kind_of(text: str) -> tuple:
    # (kind, votes, k) for one sub-question.
    from gea.indexes import embed_text
    vectors, kinds = examples()
    kind, votes = vote(embed_text([text], kind="query")[0], vectors, kinds, cfg.KIND_VOTE_K)
    return kind, votes, cfg.KIND_VOTE_K
