# Day 4 smoke test (BUILD_PLAN.md Day 4, step 4.6): the WHOLE graph on real papers, with
# the rules planner, on 5 dev questions. Not a score: the point is that real evidence --
# paragraphs, parsed tables, figures, zoomed crops -- flows through every node to an
# answer with boxes, without crashing. The answers stay clumsy until the language model
# (Days 5-6).
#
#   python scripts/smoke_real.py
#
# N1's stand-in can only tell a table or figure question when the question says "table"
# or "plot", which SPIQA questions rarely do; Day 5's model reads it properly. To exercise
# N6b and N6c now, four questions run with cfg.SUB_GOAL_MODALITY set to their reference
# kind (a testing knob, never used in evaluation); one runs as is.
# Writes results/day4_smoke.json.
import sys
import time

from gea import config as cfg
from gea.graph import build_grounded_agent_graph
from gea.safeio import atomic_write_json, read_jsonl
from gea.state import new_state

OUT = cfg.PROJECT_ROOT / "results" / "day4_smoke.json"
CASES = [("1803.03467v4#0", "table"), ("1812.06589v2#2", "table"),
         ("1803.03467v4#1", "figure"), ("1804.07931v2#0", "figure"),
         ("1804.07931v2#1", None)]


def main() -> int:
    questions = {q["qid"]: q for q in read_jsonl(cfg.PROJECT_ROOT / "eval" / "splits" / "dev.jsonl")}
    agent = build_grounded_agent_graph()
    results = []
    for qid, modality in CASES:
        q = questions[qid]
        start = time.time()
        with cfg.override(BACKEND="real", PLANNER_MODE="rules", TRACE=False, SUB_GOAL_MODALITY=modality):
            out = agent.invoke(new_state(q["question"], q["paper_id"]))
        evidence = out["evidence"]
        r = {"qid": qid, "question": q["question"], "forced_modality": modality,
             "reference": q["ref_label"], "seconds": round(time.time() - start, 1),
             "path": [e["short"] or e["node"] for e in out["events"]],
             "evidence": [f"{e.kind}/{e.label or e.source} p.{e.page} ({e.tool})" for e in evidence],
             "reference_in_buffer": any(e.label == q["ref_label"] for e in evidence),
             "table_cells": sum(len(e.cells) for e in evidence),
             "zoom_px": [list(e.crop and __import__("PIL.Image", fromlist=["Image"]).open(
                 __import__("io").BytesIO(e.crop)).size) for e in evidence if e.tool == "RegionZoom"],
             "heat_maps": sum(e.similarity is not None for e in evidence),
             "claims": len(out["claims"]), "rejected": sum(bool(c.get("rejected")) for c in out["claims"]),
             "boxes": [f"p.{b['page']} {b['what']} {b['bbox']}" for b in out.get("boxes", [])],
             "status": out.get("status"), "answer": out["answer"]}
        results.append(r)
        print(f"{qid} ({modality or 'as is'}), {r['seconds']} s: {' -> '.join(r['path'])}")
        print(f"   evidence: {r['evidence']}")
        print(f"   reference {q['ref_label']} in the buffer: {r['reference_in_buffer']};  "
              f"{r['claims']} claim(s), {r['rejected']} rejected;  boxes: {r['boxes']}")
        print(f"   answer: {r['answer'][:150]}\n")
    atomic_write_json(OUT, {"cases": results, "all_ran": True,
                            "reference_found": sum(r["reference_in_buffer"] for r in results)})
    print(f"all {len(results)} ran end to end; reference in the buffer for "
          f"{sum(r['reference_in_buffer'] for r in results)}/{len(results)}; saved {OUT.relative_to(cfg.PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
