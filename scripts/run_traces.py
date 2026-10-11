# Day 5 (BUILD_PLAN.md step 5.7): dev questions through the whole graph with the model in
# charge of N1, N2, N5 and G1, on real papers. Every trace line goes to
# notes/traces_day5.md -- to read, also during an outage -- and the facts to
# results/day5_traces.json.
#
#   bash scripts/start_llm.sh               # in its own terminal, first
#   python scripts/run_traces.py            # the first 10 dev questions
#   python scripts/run_traces.py --n 20
#
# Re-running is instant: every model reply comes back from results/llm_cache. For each
# question it also records whether N1 gave the reference's kind of evidence (table or
# figure) to a sub-goal -- the regex stand-in almost never could.
import argparse
import sys
import time

import gea.llm as llm
from gea import config as cfg
from gea.graph import build_grounded_agent_graph
from gea.safeio import atomic_write_json, atomic_write_text, read_jsonl
from gea.state import new_state

OUT_JSON = cfg.PROJECT_ROOT / "results" / "day5_traces.json"
OUT_MD = cfg.PROJECT_ROOT / "notes" / "traces_day5.md"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10, help="how many dev questions (in file order)")
    args = ap.parse_args()
    questions = read_jsonl(cfg.PROJECT_ROOT / "eval" / "splits" / "dev.jsonl")[:args.n]
    if not llm.llm_ready():
        print(f"No model server at {cfg.LLM_URL} (start it: bash scripts/start_llm.sh). Cached replies are "
              f"used; any new call falls back to the rules, and the trace says so.\n")

    agent, runs, md = build_grounded_agent_graph(), [], ["# Day 5 traces (dev questions, the model in charge)", ""]
    with cfg.override(BACKEND="real", LLM_MODE="llm", PLANNER_MODE="llm", TRACE=False):
        for q in questions:
            before = dict(llm.stats)
            start = time.time()
            out = agent.invoke(new_state(q["question"], q["paper_id"]))
            calls = {k: round(llm.stats[k] - before[k], 1) for k in llm.stats}
            lines = [f"{e['node']:6s} {e['msg']}" for e in out["events"]]
            fallbacks = sum("model unavailable" in e["msg"] or "LLM unavailable" in e["msg"] for e in out["events"])
            kinds = [g["modality"] for g in out["sub_goals"]]
            r = {"qid": q["qid"], "question": q["question"], "answer_gold": q["answer"],
                 "reference": q["ref_label"], "ref_kind": q["ref_kind"], "sub_goal_kinds": kinds,
                 "n1_kind_right": q["ref_kind"] in kinds,
                 "reference_in_buffer": any(e.label == q["ref_label"] for e in out["evidence"]),
                 "rounds": out["round"], "status": out.get("status"), "answer": out["answer"],
                 "model_calls": calls["calls"], "cached": calls["cached"], "model_seconds": calls["seconds"],
                 "fallbacks": fallbacks, "seconds": round(time.time() - start, 1), "trace": lines}
            runs.append(r)
            print(f"{q['qid']:16s} {r['seconds']:5.1f} s, {r['model_calls']:.0f} calls ({r['cached']:.0f} cached), "
                  f"N1 kinds {kinds} (reference: {q['ref_kind']}), {q['ref_label']} in buffer: "
                  f"{r['reference_in_buffer']}, rounds {r['rounds']}, {r['status']}", flush=True)
            md += [f"## {q['qid']}: {q['question']}", "",
                   f"Reference: {q['ref_label']} ({q['ref_kind']}). Gold answer: {q['answer']}", "",
                   "```text", *lines, "```", "", f"**Answer:** {out['answer']}", ""]

    n = len(runs)
    summary = {"questions": n, "prompt_version": llm.PROMPT_VERSION,
               "n1_kind_right": sum(r["n1_kind_right"] for r in runs),
               "reference_in_buffer": sum(r["reference_in_buffer"] for r in runs),
               "abstained": sum(r["status"] == "insufficient_evidence" for r in runs),
               "model_calls": sum(r["model_calls"] for r in runs), "cached": sum(r["cached"] for r in runs),
               "fallbacks": sum(r["fallbacks"] for r in runs),
               "model_seconds": round(sum(r["model_seconds"] for r in runs), 1),
               "seconds": round(sum(r["seconds"] for r in runs), 1)}
    atomic_write_json(OUT_JSON, {"summary": summary, "runs": runs})
    atomic_write_text(OUT_MD, "\n".join(md))
    print(f"\n{n} questions: N1 gave the reference's kind to a sub-goal in {summary['n1_kind_right']}, the "
          f"reference reached the buffer in {summary['reference_in_buffer']}, {summary['abstained']} abstained; "
          f"{summary['model_calls']:.0f} model calls ({summary['cached']:.0f} cached, {summary['fallbacks']} fallbacks), "
          f"{summary['seconds']} s in all")
    print(f"saved {OUT_MD.relative_to(cfg.PROJECT_ROOT).as_posix()} and {OUT_JSON.relative_to(cfg.PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
