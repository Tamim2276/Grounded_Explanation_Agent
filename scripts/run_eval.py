# Day 7.3: the evaluation runner -- one system on one split, one JSON line per question.
#
#   python scripts/run_eval.py --system full --split test          # -> results/runs/full_test.jsonl
#   python scripts/run_eval.py --system single_pass --split test   # (also: text_only)
#   python scripts/run_eval.py --system full --split test --provider gemini   # -> full_gemini_test.jsonl
#   python scripts/run_eval.py --system full --split test --limit 3          # the smoke test (7.7)
#   python scripts/run_eval.py --system full --split test --rebox            # second pass (server OFF)
#
# Power cuts: each finished question is forced to disk (append_jsonl) before the next one
# starts, and a restart skips every qid already in the file. Every model reply is cached,
# so the interrupted question is redone in seconds, identically.
# The model unreachable: a question during which the server (or Gemini) could not be
# reached is thrown away, the runner WAITS until the model answers, and redoes it -- the
# rules' stand-in answers never enter the results. (A bad reply is different: that is the
# system's own behaviour, recorded as "fallbacks".)
# The test split: no question or answer text is printed (it stays unread until Day 8), and
# the runner refuses to start unless the settings match eval/frozen_config.json.
# --rebox (Day 7, after scripts/claim_vectors.py): replays, from the cache, the questions
# whose figure boxes came from the question's map, so each claim gets its own map's box.
import argparse
import json
import sys
import time

import gea.llm as llm
from gea import config as cfg
from gea.graph import build_graph
from gea.indexes import query_cache_path
from gea.runs import frozen_differences, model_name, plain, record
from gea.safeio import append_jsonl, atomic_write_text, read_jsonl
from gea.state import new_state

RUNS = cfg.PROJECT_ROOT / "results" / "runs"
SYSTEMS = ("full", "single_pass", "text_only")
# Ablations: a run that differs from the frozen settings in exactly these, named in its file and
# its records. "strict": the first test run (2026-10-15) -- G1 alone decides, N5 from round 1.
ABLATIONS = {"strict": {"AFTER_BUDGET": "abstain", "FIRST_ROUND": "planner"}}
MAX_REDOS = 5            # then stop: the model answers "ready" but refuses every request (Gemini 429:
                         # a used-up daily quota) -- waiting longer would only loop


def run_one(agent, q: dict) -> tuple:
    # (graph output, seconds, model-call counts) of one question
    before, start = dict(llm.stats), time.time()
    out = agent.invoke(new_state(q["question"], q["paper_id"]))
    return out, round(time.time() - start, 1), {k: llm.stats[k] - before[k] for k in llm.stats}


def wait_for_model() -> None:
    if llm.llm_ready():
        return
    print(f"{time.strftime('%H:%M')} the model ({cfg.LLM_PROVIDER}) does not answer; waiting ...", flush=True)
    while not llm.llm_ready():
        time.sleep(30)
    print(f"{time.strftime('%H:%M')} the model answers again", flush=True)


def progress(done: int, total: int, seconds_left: float) -> str:
    # "[ 12/150   8%  ~1h 25m left] " -- the time left from this session's pace
    left = f"{int(seconds_left // 3600)}h {int(seconds_left % 3600 // 60):02d}m" if seconds_left >= 3600 \
        else f"{int(seconds_left // 60)}m"
    return f"[{done:3d}/{total}  {100 * done // total:3d}%  ~{left} left] "


def show(q: dict, r: dict, test: bool) -> str:
    kept = sum(not c["rejected"] for c in r["claims"])
    line = (f"{q['qid']:18s} {r['seconds']:6.1f} s  {r['status'] or 'error':28s} claims {kept}/{len(r['claims'])}, "
            f"boxes {len(r['boxes'])}, calls {r['model_calls']} ({r['cached']} cached)")
    return line if test else f"{line}  |  {(r['answer'] or '')[:60]}"


def rebox(out_file, questions: dict, agent) -> int:
    # Replay the questions with a map box; keep the new boxes only if EVERY reply was cached
    # (the language model is off now -- anything else would change the answer).
    records, changed = read_jsonl(out_file), 0
    for r in records:
        if r["qid"] not in questions or not any("(patch map)" in b["what"] for b in r["boxes"]):
            continue
        out, _, calls = run_one(agent, questions[r["qid"]])
        if calls["calls"] != calls["cached"]:
            print(f"{r['qid']}: a reply was not cached -- boxes kept as they were")
            continue
        boxes = plain(out.get("boxes", []))
        if boxes != r["boxes"]:
            r["boxes"], r["reboxed"] = boxes, True
            changed += 1
    atomic_write_text(out_file, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(f"reboxed {changed} question(s); {sum('(claim map)' in b['what'] for r in records for b in r['boxes'])} "
          f"box(es) now from a claim's own map")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", choices=SYSTEMS, default="full")
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--provider", choices=("local", "gemini"), default="local")
    ap.add_argument("--limit", type=int, help="only the first N questions (the smoke test)")
    ap.add_argument("--out", help="default results/runs/<system>[_gemini]_<split>.jsonl")
    ap.add_argument("--rebox", action="store_true", help="second pass: claim-map boxes, from the cache")
    ap.add_argument("--ablation", choices=sorted(ABLATIONS), help="a documented variant of the frozen settings")
    args = ap.parse_args()
    name = (args.system + ("_gemini" if args.provider == "gemini" else "")
            + (f"_{args.ablation}" if args.ablation else ""))
    out_file = RUNS / (args.out or f"{name}_{args.split}.jsonl")
    questions = read_jsonl(cfg.PROJECT_ROOT / "eval" / "splits" / f"{args.split}.jsonl")[:args.limit]
    test = args.split == "test"

    missing = sum(not query_cache_path(q["question"]).exists() for q in questions)
    if missing:          # without them the region choice would silently lose ColQwen2's score
        print(f"{missing} question(s) have no cached ColQwen2 vectors: run, with the model server OFF,\n"
              f"  python scripts/cache_question_vectors.py --split {args.split}")
        return 1
    settings = dict(BACKEND="real", LLM_MODE="llm", PLANNER_MODE="llm", TRACE=False,
                    VARIANT=args.system, LLM_PROVIDER=args.provider, **ABLATIONS.get(args.ablation, {}))
    with cfg.override(**settings):
        if test and (problems := frozen_differences(tuple(ABLATIONS.get(args.ablation, {})))):
            print("The settings differ from eval/frozen_config.json -- the test run must use the frozen ones:")
            print("\n".join(f"  {p}" for p in problems))
            return 1
        agent = build_graph(args.system)
        if args.rebox:
            return rebox(out_file, {q["qid"]: q for q in questions}, agent)

        done = {r["qid"] for r in read_jsonl(out_file)}
        todo = [q for q in questions if q["qid"] not in done]
        finished = len(done & {q['qid'] for q in questions})
        print(f"{name} on {args.split}: {finished} done, {len(todo)} to go "
              f"-> {out_file.relative_to(cfg.PROJECT_ROOT).as_posix()}  (model {model_name()}, prompt "
              f"{llm.PROMPT_VERSION})", flush=True)
        started = time.time()
        for n, q in enumerate(todo, 1):
            redos = 0
            while True:
                try:
                    out, seconds, calls = run_one(agent, q)
                except MemoryError:
                    raise                                   # the PC, not the system: stop; a restart resumes
                except Exception as e:                      # a crash in the system is a result: record it
                    r = {"qid": q["qid"], "paper_id": q["paper_id"], "system": args.system,
                         "provider": args.provider, "model": model_name(), "status": "error",
                         "error": f"{type(e).__name__}: {e}"[:500], "claims": [], "boxes": [],
                         "answer": None, "seconds": 0.0, "model_calls": 0, "cached": 0}
                    break
                if calls["unreachable"]:                    # thrown away: never record stand-in answers
                    redos += 1
                    if redos > MAX_REDOS:                   # it answers but keeps refusing: a used-up quota
                        print(f"{q['qid']}: the model refused {MAX_REDOS} times in a row (a quota?) -- stopped; "
                              f"run this again later to continue", flush=True)
                        return 2
                    print(f"{q['qid']}: the model could not be reached during this question -- redo", flush=True)
                    time.sleep(30)                          # a rate limit (429) needs a pause too
                    wait_for_model()
                    continue
                r = record(q, out, seconds, calls, args.system) | ({"ablation": args.ablation} if args.ablation else {})
                break
            append_jsonl(out_file, r)                       # on disk before the next question starts
            print(progress(finished + n, len(questions), (time.time() - started) / n * (len(todo) - n))
                  + (show(q, r, test) if r["status"] != "error" else f"{q['qid']}: ERROR {r['error'][:120]}"), flush=True)

    records = [r for r in read_jsonl(out_file) if r["qid"] in {q["qid"] for q in questions}]
    statuses = {s: sum(r["status"] == s for r in records) for s in sorted({r["status"] for r in records})}
    print(f"\n{name} on {args.split}: {len(records)}/{len(questions)} questions in the file; {statuses}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
