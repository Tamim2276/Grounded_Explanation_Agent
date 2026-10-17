# Day 8.3: the accuracy judge. For every published answer of a run, the model compares it with
# SPIQA's gold answer and says correct or wrong, with a reason. A question with no published
# answer is recorded as abstained, without a model call.
#
#   python scripts/judge.py                                  # the five test runs
#   python scripts/judge.py full_test single_pass_test       # some of them
#   python scripts/judge.py full_dev_v6 full_dev_v7 full_dev_v8   # the dev rehearsal
#
# Writes results/judged/<run>.jsonl, one line per question; a restart skips judged questions
# and every model call is cached. When the model cannot be reached the run stops and nothing
# is written for that question: run it again later. Prints no question or answer text.
import argparse
import sys
import time

from gea import config as cfg
from gea import llm
from gea.metrics import answer_text, outcome, published
from gea.runs import model_name
from gea.safeio import append_jsonl, read_jsonl

JUDGE_VERSION = "j1"      # raise it when SYSTEM or SCHEMA change (notes/prompts.md)
TEST_RUNS = ("full_test", "full_strict_test", "single_pass_test", "text_only_test", "full_gemini_test")
RUNS_DIR = cfg.PROJECT_ROOT / "results" / "runs"
OUT_DIR = cfg.PROJECT_ROOT / "results" / "judged"

SYSTEM = """You grade answers to questions about scientific papers.
You get a question, the reference answer written by the dataset's authors, and a system's answer.
Decide whether the system's answer is correct with respect to the reference answer.

Correct: the system's answer contains the main point of the reference answer -- the value, the
name, the trend or the comparison the question asks for -- and states nothing that contradicts
the reference. Extra details that do not contradict the reference are fine. Wording, rounding,
units and level of detail may differ. An answer that gives the main point but leaves out
secondary details is correct.
Wrong: the system's answer gives a different value or name, the opposite trend, a claim that
contradicts the reference, or it is so vague or off-topic that it does not answer the question.

Judge only against the reference answer, not against your own knowledge of the paper.
First write the reason in at most two sentences (what the reference says, what the system
says), then the verdict."""

# The reason comes before the verdict: llama.cpp fills the fields in this order, and a verdict
# written first is only justified afterwards (the G1 lesson, Day 7).
SCHEMA = {"type": "object",
          "properties": {"reason": {"type": "string", "maxLength": 400}, "correct": {"type": "boolean"}},
          "required": ["reason", "correct"]}


def judge(record: dict) -> dict:
    user = (f"Question: {record['question']}\n\nReference answer: {record['gold']}\n\n"
            f"System's answer: {answer_text(record)}")
    reply = llm.call_llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                         SCHEMA, "judge")
    return {"correct": bool(reply["correct"]), "reason": str(reply["reason"])}


def wait_for_model(minutes: int = 10) -> bool:
    # The server may still be loading (about a minute after start_llm.sh).
    for _ in range(minutes * 6):
        if llm.llm_ready():
            return True
        time.sleep(10)
    return False


def judge_run(run: str) -> bool:
    records = read_jsonl(RUNS_DIR / f"{run}.jsonl")
    out = OUT_DIR / f"{run}.jsonl"
    done = {r["qid"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in records if r["qid"] not in done]
    print(f"{run}: {len(done)} judged, {len(todo)} to go -> {out.relative_to(cfg.PROJECT_ROOT)}", flush=True)
    for n, r in enumerate(todo, len(done) + 1):
        verdict, result = {"correct": None, "reason": None}, None
        if published(r):
            try:
                verdict = judge(r)
            except OSError:
                print(f"{r['qid']}: the judge model could not be reached -- stopped; run this again", flush=True)
                return False
            except ValueError as e:                    # a malformed reply: kept, counted, not guessed
                verdict, result = {"correct": None, "reason": f"judge error: {e}"}, "judge_error"
        row = {"qid": r["qid"], "run": run, "outcome": result or outcome(r, verdict["correct"]),
               "best_effort": bool(r.get("best_effort")), "status": r["status"], **verdict,
               "judge": model_name(), "judge_version": JUDGE_VERSION}
        append_jsonl(out, row)
        print(f"[{n:>3}/{len(records)}] {r['qid']:<20} {row['outcome']}", flush=True)
    rows = read_jsonl(out)
    counts = {k: sum(x["outcome"] == k for x in rows) for k in ("correct", "wrong", "abstained", "judge_error")}
    print(f"{run}: {len(rows)}/{len(records)} judged; {counts}\n", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", default=TEST_RUNS, help="run file names in results/runs/, no .jsonl")
    parser.add_argument("--provider", default="local", choices=("local",),
                        help="the judge is the local model (a second judge would get its own files)")
    args = parser.parse_args()
    cfg.LLM_PROVIDER = args.provider
    if not wait_for_model():
        sys.exit("the model server does not answer -- start it: bash scripts/start_llm.sh")
    for run in args.runs:
        if not judge_run(run):
            sys.exit(1)


if __name__ == "__main__":
    main()
