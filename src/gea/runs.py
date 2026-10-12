# The evaluation's records (BUILD_PLAN.md Day 7): one question of one system as one JSON line,
# and the frozen settings the test run must match.
#
# A record holds everything Day 8's scoring and the website (Days 11-13) need -- the claims
# with their cites, every box with its page size, G1's and G2's verdicts with reasons, the
# path through the graph, the cost -- and no arrays or image bytes.
import hashlib
import json
import subprocess
from datetime import datetime

import gea.llm as llm
from gea import config as cfg

FROZEN = cfg.PROJECT_ROOT / "eval" / "frozen_config.json"
SPLITS = cfg.PROJECT_ROOT / "eval" / "splits"
# Switches the runner sets itself, paths, and the stub's knobs: not part of what is frozen.
NOT_FROZEN = {"PROJECT_ROOT", "DATA_DIR", "MODELS_DIR", "LLM_CACHE_DIR", "TRACE", "BACKEND", "VARIANT",
              "LLM_MODE", "PLANNER_MODE", "LLM_PROVIDER", "LLM_URL", "GEMINI_URL",
              "HALLUCINATE_ONCE", "HALLUCINATE_ALWAYS", "TABLE_CLAIM_STYLE"}


def plain(value):
    # JSON-safe: tuples become lists, everything else as it is.
    return json.loads(json.dumps(value, default=str))


def model_name() -> str:
    return cfg.GEMINI_MODEL if cfg.LLM_PROVIDER == "gemini" else cfg.LLM_MODEL


def record(q: dict, out: dict, seconds: float, calls: dict, system: str) -> dict:
    # One finished question, website-ready. A withheld claim carries G2's reason.
    reasons = {f["claim"]: f["reason"] for f in out.get("faith_failures") or []}
    dag = out.get("dag_trace") or {}
    claims = [{"text": c["text"], "conclusion": bool(c.get("conclusion")), "rejected": bool(c.get("rejected")),
               **({"reason": reasons.get(i)} if c.get("rejected") else {}),
               "cites": plain(c["cites"])} for i, c in enumerate(out.get("claims", []))]
    return {"qid": q["qid"], "paper_id": q["paper_id"], "question": q["question"], "gold": q["answer"],
            "reference": q["ref_label"], "ref_kind": q["ref_kind"],
            "system": system, "provider": cfg.LLM_PROVIDER, "model": model_name(),
            "prompt_version": llm.PROMPT_VERSION,
            "status": out.get("status"), "answer": out.get("answer"), "claims": claims,
            "boxes": plain(out.get("boxes", [])),
            "evidence": [{"id": e.id, "kind": e.kind, "label": e.label or e.source, "page": e.page,
                          "tool": e.tool, "bbox": plain(e.bbox), "page_size": plain(e.page_size)}
                         for e in out.get("evidence", [])],
            "sub_goals": plain(out.get("sub_goals", [])),
            "g1": plain(out.get("sufficiency") or {}), "g2_last_failures": plain(out.get("faith_failures") or []),
            "rounds": out.get("round"), "drafts": out.get("drafts"), "rewrites": out.get("rewrite_count", 0),
            "tool_calls": dag.get("tool_calls"), "g1_rejections": dag.get("g1_rejections"),
            "g2_rejections": dag.get("g2_rejections"), "dag_path": dag.get("path"),
            "reference_in_buffer": any(e.label == q["ref_label"] for e in out.get("evidence", [])),
            "model_calls": calls["calls"], "cached": calls["cached"],
            "fallbacks": calls["refused"] + calls["bad_replies"],    # nodes that used their rules
            "seconds": seconds}


def file_hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def frozen_settings() -> dict:
    # Everything that changes an answer: every setting except the switches above, the prompt
    # version, the model files and the exact question splits.
    settings = {k: plain(v) for k, v in sorted(vars(cfg).items())
                if k.isupper() and k not in NOT_FROZEN and not callable(v)}
    settings["PROMPT_VERSION"] = llm.PROMPT_VERSION
    settings["model_files"] = {p.name: p.stat().st_size for p in sorted(cfg.MODELS_DIR.rglob("*.gguf"))}
    settings["splits"] = {p.name: file_hash(p) for p in sorted(SPLITS.glob("*.jsonl"))}
    return settings


def freeze() -> dict:
    # The settings plus the code version: the commit, and whether files were still uncommitted
    # (then the checkpoint right after the freeze holds the frozen code).
    git = lambda *a: subprocess.run(["git", *a], cwd=cfg.PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    return {"frozen_at": datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain")), "settings": frozen_settings()}


def frozen_differences() -> list:
    # What differs from eval/frozen_config.json, in words ([] = the settings match).
    if not FROZEN.exists():
        return ["eval/frozen_config.json does not exist yet (python scripts/freeze_config.py)"]
    frozen, now = json.loads(FROZEN.read_text(encoding="utf-8"))["settings"], frozen_settings()
    return [f"{k}: frozen {frozen.get(k)!r}, now {now.get(k)!r}"
            for k in sorted(set(frozen) | set(now)) if frozen.get(k) != now.get(k)]
