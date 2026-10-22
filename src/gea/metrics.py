# Day 8: the thesis's scores. They read the run files (results/runs/), the judge's verdicts
# (results/judged/) and the hand annotations (eval/); nothing here calls a model.
#
#   outcome  correct / wrong / abstained per question -- never accuracy alone
#   RF       Retrieval Faithfulness: the share of the visual items fetched that are the reference
#   hit      the reference table or figure was in the buffer at all
#   AES      Agent Efficiency Score: hops needed / search-tool calls made (capped at 1)
#   CGS      Cross-modal Grounding Score: IoU of the proof box with the hand-drawn answer box
#   MAP      Modality Attribution Precision: the share of published claims citing the right
#            source (hand-marked on the 50 annotated questions)
import numpy as np

from gea.proofs import iou

FLAG = " (The evidence found may be incomplete.)"           # N12's mark on a best-effort answer
SEARCH_TOOLS = ("TextRetriever", "TableRetriever", "FigureRetriever")   # N6a-N6c; N7 (zoom) is automatic
SEED = 42


def published(record: dict) -> bool:
    # Answered = at least one claim reached the reader. An abstention, or every claim withheld
    # by G2 ("nothing can be stated"), is not an answer.
    return any(not c["rejected"] for c in record["claims"])


def answer_text(record: dict) -> str:
    # The published answer without the best-effort mark: the judge grades what is claimed,
    # not the warning that comes with it.
    return (record.get("answer") or "").replace(FLAG, "").strip()


def outcome(record: dict, correct) -> str:
    # One of three, from a run record and the judge's verdict (None when not judged).
    if not published(record):
        return "abstained"
    return "correct" if correct else "wrong"


def visual_labels(record: dict) -> set:
    # The tables and figures in the buffer, each once (a zoom is the figure it enlarges).
    return {e["label"] for e in record["evidence"] if e["kind"] in ("table", "figure")}


def rf(labels, gold: str):
    # None when nothing visual was retrieved (text-only): the score is undefined, not 0.
    labels = list(labels)
    return sum(label == gold for label in labels) / len(labels) if labels else None


def search_calls(record: dict) -> int:
    # Calls of N6a-N6c from the DAG trace. The record's own tool_calls also counts N7.
    return sum(step in SEARCH_TOOLS for step in (record.get("dag_path") or "").split(" -> "))


def aes(needed_hops: int, tool_calls: int) -> float:
    return min(1.0, needed_hops / tool_calls) if tool_calls else 0.0


def cgs(boxes: list, gold: list) -> float:
    # The best IoU between any proof box and any hand-drawn box on the same page: does a proof
    # point at the answer? `boxes` as in a run record ({page, bbox}); `gold` as (page, bbox)
    # pairs. No box (an abstention, or no box drawn) scores 0.
    return max((iou(tuple(b["bbox"]), tuple(g)) for b in boxes for page, g in gold if b["page"] == page),
               default=0.0)


def reference_attribution(record: dict):
    # MAP, automatic stand-in (the hand check was skipped for time, 2026-10-22): the share of
    # published claims that cite the question's reference table or figure. It undercounts claims
    # that rightly cite supporting text. None when nothing was published.
    labels = {e["id"]: e["label"] for e in record["evidence"]}
    claims = [c for c in record["claims"] if not c["rejected"]]
    if not claims:
        return None
    return sum(any(labels.get(cite.get("evidence")) == record["reference"] for cite in c["cites"] or [])
               for c in claims) / len(claims)


def center_inside(box, gold_box) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return gold_box[0] <= cx <= gold_box[2] and gold_box[1] <= cy <= gold_box[3]


def pointing(boxes: list, gold: list) -> float:
    # The pointing game: 1 when the centre of any proof box lies inside a hand-drawn box on the
    # same page. Added after the IoU results (2026-10-22): IoU punishes a correct proof whose box
    # is drawn at another size -- a whole table cell against a hand box tight around the number
    # scores 0.43, a correct cell inside a hand box around the row 0.18.
    return float(any(center_inside(b["bbox"], g) for b in boxes for page, g in gold if b["page"] == page))


def pointing_share(boxes: list, gold: list):
    # The share of a question's proof boxes whose centre lies inside a hand-drawn box: how
    # focused the proofs are. None without boxes.
    if not boxes:
        return None
    return sum(any(center_inside(b["bbox"], g) for page, g in gold if b["page"] == page) for b in boxes) / len(boxes)


def bootstrap_mean(a, n: int = 10_000, seed: int = SEED) -> tuple:
    # The mean of per-question scores and its 95% interval: resample the questions with
    # replacement n times.
    a = np.asarray(a, float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n, len(a)))
    means = a[idx].mean(axis=1)
    return float(a.mean()), tuple(float(v) for v in np.percentile(means, [2.5, 97.5]))


def bootstrap_diff(a, b, n: int = 10_000, seed: int = SEED) -> tuple:
    # a, b: per-question scores of two systems, same questions, same order. The mean
    # difference a - b and its 95% interval; an interval above 0 means a is better beyond luck.
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float((a - b).mean()), tuple(float(v) for v in np.percentile(diffs, [2.5, 97.5]))


def cohen_kappa(a: list, b: list) -> float:
    # Agreement of two graders corrected for the agreement expected by chance: 1 = perfect,
    # 0 = no better than chance.
    labels = sorted(set(a) | set(b))
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    expected = sum((a.count(k) / len(a)) * (b.count(k) / len(b)) for k in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0
