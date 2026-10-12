# The local language model: Qwen2.5-VL-7B served by llama.cpp's llama-server on
# the B580 (BUILD_PLAN.md Day 1 installs it, Day 5 wires it into the nodes).
#
# Every call goes through call_llm(): temperature 0, a fixed seed, and a JSON schema that
# llama.cpp enforces token by token, so a reply is always a valid object of the asked
# shape. Every reply is cached in results/llm_cache/ (in git) under a fingerprint of the
# exact request: asking again -- a rerun, or catching up after a power cut -- is instant,
# free and identical, and the thesis numbers can be traced to these exact replies.
import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

from gea import config as cfg
from gea.safeio import atomic_write_json
from gea.state import AgentState

PROMPT_VERSION = "v7"     # raise it whenever a prompt changes (notes/prompts.md): old entries stop matching
last_call = {}            # {"name", "cached", "seconds"} of the most recent call, for the traces
# Since the session started (scripts report it). A run counts what went wrong apart:
#   unreachable  no server, a timeout, 429 (rate limit) or 5xx -- the question must be REDONE
#                (scripts/run_eval.py), never recorded with the rules' stand-in answers;
#   refused      another 4xx (e.g. the request was too long) -- the node used its rules;
#   bad_replies  a reply without the required fields -- the node used its rules.
# The last two are behaviour of the system under test and stay in the results.
stats = {"calls": 0, "cached": 0, "seconds": 0.0, "unreachable": 0, "refused": 0, "bad_replies": 0}
RETRY_WAITS = (10, 20, 40, 60)   # seconds before retrying one call that got 429 or 5xx (then it counts
                                 # as unreachable and the runner redoes the whole question later)


def gemini_key() -> str:
    # GEMINI_API_KEY from the environment, else from the project's .env (which git ignores).
    # Never printed. RuntimeError, not OSError: a missing key must stop a run, not make every
    # node quietly fall back to its rules.
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    env = cfg.PROJECT_ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key or key == "paste-your-key-here":
        raise RuntimeError("no Gemini key: paste it into .env (GEMINI_API_KEY=...) or set the environment variable")
    return key


def endpoint() -> tuple:
    # (base URL, model name, extra headers) of cfg.LLM_PROVIDER.
    if cfg.LLM_PROVIDER == "local":
        return cfg.LLM_URL, cfg.LLM_MODEL, {}
    if cfg.LLM_PROVIDER == "gemini":
        return cfg.GEMINI_URL, cfg.GEMINI_MODEL, {"Authorization": f"Bearer {gemini_key()}"}
    raise RuntimeError(f"unknown LLM_PROVIDER {cfg.LLM_PROVIDER!r}")


def llm_ready() -> bool:
    # True if the model answers: llama-server at LLM_URL, or Gemini (network and key).
    url, _, headers = endpoint()
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{url}/models", headers=headers), timeout=10) as reply:
            return reply.status == 200
    except OSError:                   # not running, refused, timed out, offline
        return False


def call_llm(messages: list, schema: dict, name: str, images: tuple = ()) -> dict:
    # One model call with a forced JSON answer: `messages` in the chat format, `schema`
    # the answer's shape, `name` a label for the cache and traces, `images` PNG bytes
    # attached to the last user message. Raises OSError when the server cannot be
    # reached and ValueError on a reply without the required fields -- the calling node
    # then falls back to its rules and says so in the trace.
    if images:
        parts = [{"type": "text", "text": messages[-1]["content"]}]
        parts += [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}
                  for png in images]
        messages = messages[:-1] + [{"role": "user", "content": parts}]
    url, model, headers = endpoint()
    body = {"model": model, "temperature": 0, "seed": cfg.SEED, "messages": messages,
            "max_tokens": cfg.LLM_MAX_TOKENS,
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}}
    if cfg.LLM_PROVIDER == "gemini":
        del body["seed"]          # Gemini's endpoint rejects it (400); its replies are cached all the same
    key = hashlib.sha256(json.dumps([PROMPT_VERSION, body], sort_keys=True).encode("utf-8")).hexdigest()
    path = cfg.LLM_CACHE_DIR / key[:2] / f"{key}.json"          # a sub-folder per 2 hex digits
    stats["calls"] += 1
    if cfg.LLM_CACHE and path.exists():
        stats["cached"] += 1
        last_call.update(name=name, cached=True, seconds=0.0)
        return json.loads(path.read_text(encoding="utf-8"))["answer"]

    request = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", **headers})
    start = time.perf_counter()
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            with urllib.request.urlopen(request, timeout=300) as reply:
                out = json.load(reply)
            break
        except urllib.error.HTTPError as e:
            busy = e.code == 429 or e.code >= 500
            if busy and attempt < len(RETRY_WAITS):        # Gemini's "high demand" 503 passes in seconds
                time.sleep(RETRY_WAITS[attempt])
                continue
            stats["unreachable" if busy else "refused"] += 1
            raise
        except OSError:                                   # no server at all: no point in waiting here
            stats["unreachable"] += 1
            raise
    try:
        answer = json.loads(out["choices"][0]["message"]["content"])
        missing = [k for k in schema.get("required", []) if k not in answer]
        if missing:
            raise ValueError(f"{name}: reply lacks {missing}")
    except (ValueError, KeyError, IndexError, TypeError):
        stats["bad_replies"] += 1
        raise
    seconds = round(time.perf_counter() - start, 1)
    shown = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else
              " ".join(p.get("text", "[image]") for p in m["content"])} for m in messages]
    if cfg.LLM_CACHE:
        atomic_write_json(path, {"name": name, "prompt_version": PROMPT_VERSION, "model": model,
                                 "seconds": seconds, "images": len(images), "usage": out.get("usage", {}),
                                 "messages": shown, "answer": answer})    # never a half-written entry
    last_call.update(name=name, cached=False, seconds=seconds)
    stats["seconds"] += seconds
    return answer


def how_answered() -> str:
    # " (model, 4.3 s)" or " (model, cached)" for the trace line of the node that just called
    if not last_call:
        return ""
    return " (model, cached)" if last_call["cached"] else f" (model, {last_call['seconds']} s)"


# --- N5: the planner ----------------------------------------------------------------

TOOL_TEXT = {"text": "'text' searches paragraphs", "table": "'table' searches tables",
             "figure": "'figure' searches charts"}

def llm_plan(state: AgentState) -> dict:
    # Ask the local model which tool to call next.
    # Returns {"tool", "query", "reason"}. Raises OSError if the server cannot be
    # reached, ValueError on a reply that is not the required JSON -- the caller
    # (planner.propose_action) decides what to do.
    goals = "\n".join(f"{i}. [{g['modality']}] {g['text']}" for i, g in enumerate(state["sub_goals"], 1))
    tried = ", ".join(f"sub-goal {g + 1} with {t}" for g, t in state["tried"]) or "nothing yet"
    missing = (state.get("sufficiency") or {}).get("missing") or "nothing checked yet"
    tools = ", ".join(TOOL_TEXT[t] for t in cfg.tools())        # the text-only baseline offers one
    system = ("You plan retrieval for questions about one scientific paper. Pick ONE tool: "
              f"{tools}. "
              "Do not repeat a tool already tried for the same sub-goal. Give a one-sentence "
              "reason, then the tool, then a short search query.")
    user = (f"Question: {state['question']}\nSub-goals:\n{goals}\n"
            f"Already tried: {tried}\nStill missing: {missing}")
    schema = {"type": "object",
              "properties": {"reason": {"type": "string"},
                             "tool":   {"type": "string", "enum": list(cfg.tools())},
                             "query":  {"type": "string"}},
              "required": ["reason", "tool", "query"]}
    answer = call_llm([{"role": "system", "content": system}, {"role": "user", "content": user}], schema, "plan")
    return {"tool": answer["tool"], "query": answer["query"], "reason": answer["reason"],
            "seconds": last_call["seconds"]}
