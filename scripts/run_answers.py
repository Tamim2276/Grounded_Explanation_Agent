# Day 6, step 6.5: real grounded answers on dev questions -- the whole graph with the model
# in N1, N2, N5, G1, N9 and G2's support check, on real papers. Saves every answer in the
# website-ready shape (claims with their cites, boxes with page size, G1/G2 verdicts,
# path, model and prompt version) and draws proof images for the first questions that
# have boxes.
#
#   bash scripts/start_llm.sh               # first, in its own terminal
#   python scripts/run_answers.py           # 10 dev questions
#   python scripts/run_answers.py --n 20 --proofs 3
#   python scripts/run_answers.py --n 20 --tag day7_dev   # Day 7.1: all dev, own files
#
# Writes results/<tag>_answers.json, notes/answers_<tag>.md and data/debug/<tag>/proof_*.png
# (tag "day6" by default).
# Re-running is instant: every model reply comes back from results/llm_cache.
import argparse
import sys
import time

import pymupdf

import gea.llm as llm
from gea import config as cfg
from gea.corpus import get_corpus
from gea.graph import build_grounded_agent_graph
from gea.safeio import atomic_write_json, atomic_write_text, read_jsonl
from gea.state import new_state
from gea.viz import draw_proof



def cite_text(c: dict) -> str:
    if "cell" in c:
        return f"{c['evidence']} cell {c['cell'][0]} / {c['cell'][1]} = {c['value']}"
    if "quote" in c:
        return f"{c['evidence']} quote \"{c['quote'][:90]}\""
    return f"{c['evidence']} observation \"{c['observation'][:90]}\""


def record(q: dict, out: dict, seconds: float, calls: dict) -> dict:
    # One answer, website-ready: no arrays, no image bytes.
    return {"qid": q["qid"], "paper_id": q["paper_id"], "question": q["question"], "gold": q["answer"],
            "reference": q["ref_label"], "ref_kind": q["ref_kind"], "status": out.get("status"),
            "answer": out["answer"], "rounds": out["round"], "drafts": out.get("drafts"),
            "rewrites": out.get("rewrite_count", 0),
            "claims": [{"text": c["text"], "conclusion": bool(c.get("conclusion")), "rejected": bool(c.get("rejected")),
                        "cites": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in x.items()} for x in c["cites"]]}
                       for c in out["claims"]],
            "boxes": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in b.items()} for b in out.get("boxes", [])],
            "evidence": [{"id": e.id, "kind": e.kind, "label": e.label or e.source, "page": e.page, "tool": e.tool}
                         for e in out["evidence"]],
            "g1": {k: v for k, v in (out.get("sufficiency") or {}).items()},
            "g2_last_failures": out.get("faith_failures", []),
            "path": (out.get("dag_trace") or {}).get("path"),
            "reference_in_buffer": any(e.label == q["ref_label"] for e in out["evidence"]),
            "model": cfg.LLM_MODEL, "prompt_version": llm.PROMPT_VERSION,
            "model_calls": calls["calls"], "cached": calls["cached"], "seconds": seconds}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--proofs", type=int, default=3)
    ap.add_argument("--tag", default="day6", help="names the output files (results/<tag>_answers.json, ...)")
    args = ap.parse_args()
    out_file = cfg.PROJECT_ROOT / "results" / f"{args.tag}_answers.json"
    notes_file = cfg.PROJECT_ROOT / "notes" / f"answers_{args.tag}.md"
    proofs_dir = cfg.DATA_DIR / "debug" / args.tag
    if not llm.llm_ready():
        print(f"No model server at {cfg.LLM_URL} (bash scripts/start_llm.sh): cached replies are used, "
              f"new calls fall back to the rules and the trace says so.\n")
    questions = read_jsonl(cfg.PROJECT_ROOT / "eval" / "splits" / "dev.jsonl")[:args.n]
    agent, runs, md, drawn = build_grounded_agent_graph(), [], [f"# Real grounded answers (dev, {args.tag})", ""], 0
    proofs_dir.mkdir(parents=True, exist_ok=True)
    with cfg.override(BACKEND="real", LLM_MODE="llm", PLANNER_MODE="llm", TRACE=False):
        for q in questions:
            before, start = dict(llm.stats), time.time()
            out = agent.invoke(new_state(q["question"], q["paper_id"]))
            r = record(q, out, round(time.time() - start, 1), {k: llm.stats[k] - before[k] for k in llm.stats})
            runs.append(r)
            kept = [c for c in out["claims"] if not c.get("rejected")]
            print(f"{q['qid']:16s} {r['seconds']:5.1f} s  {r['status']:28s} claims {len(kept)}/{len(out['claims'])}, "
                  f"drafts {r['drafts']}, boxes {len(r['boxes'])}  |  {out['answer'][:70]}", flush=True)
            md += [f"## {q['qid']}: {q['question']}", "", f"**Gold:** {q['answer']} ({q['ref_label']})", "",
                   f"**Answer** ({r['status']}): {out['answer']}", ""]
            md += [f"- {'~~' if c.get('rejected') else ''}{c['text']}{'~~' if c.get('rejected') else ''} "
                   f"[{'; '.join(cite_text(x) for x in c['cites'])}]" for c in out["claims"]]
            md += [""] + [f"  - box p.{b['page']} {b['what']} {b['bbox']}" for b in out.get("boxes", [])] + [""]
            if drawn < args.proofs and out.get("boxes"):
                with pymupdf.open(get_corpus(q["paper_id"])["pdf"]) as doc:
                    for k, image in enumerate(draw_proof(out, doc)):
                        image.save(proofs_dir / f"proof_{q['qid'].replace('#', '_')}_{k + 1}.png")
                drawn += 1
    n = len(runs)
    summary = {"questions": n, "prompt_version": llm.PROMPT_VERSION,
               "answered": sum(r["status"] == "ok" for r in runs),
               "abstained": sum(r["status"] == "insufficient_evidence" for r in runs),
               "claims_dropped_runs": sum(r["status"] == "unsupported_claims_dropped" for r in runs),
               "claims": sum(len(r["claims"]) for r in runs),
               "claims_kept": sum(not c["rejected"] for r in runs for c in r["claims"]),
               "rewrites": sum(r["rewrites"] for r in runs),
               "reference_in_buffer": sum(r["reference_in_buffer"] for r in runs),
               "model_calls": sum(r["model_calls"] for r in runs), "seconds": round(sum(r["seconds"] for r in runs), 1)}
    atomic_write_json(out_file, {"summary": summary, "runs": runs})
    atomic_write_text(notes_file, "\n".join(md))
    print(f"\n{summary}\nsaved {out_file.relative_to(cfg.PROJECT_ROOT).as_posix()}, "
          f"{notes_file.relative_to(cfg.PROJECT_ROOT).as_posix()}, {drawn} proof image set(s) in data/debug/{args.tag}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
