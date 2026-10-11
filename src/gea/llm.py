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
import time
import urllib.request

from gea import config as cfg
from gea.safeio import atomic_write_json
from gea.state import AgentState

PROMPT_VERSION = "v3"     # raise it whenever a prompt changes (notes/prompts.md): old entries stop matching
last_call = {}            # {"name", "cached", "seconds"} of the most recent call, for the traces
stats = {"calls": 0, "cached": 0, "seconds": 0.0}     # since the session started (scripts report it)


def llm_ready() -> bool:
    # True if a llama.cpp server answers at LLM_URL.
    try:
        with urllib.request.urlopen(f"{cfg.LLM_URL}/models", timeout=3) as reply:
            return reply.status == 200
    except OSError:                   # not running, refused, timed out
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
    body = {"model": cfg.LLM_MODEL, "temperature": 0, "seed": cfg.SEED, "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}}
    key = hashlib.sha256(json.dumps([PROMPT_VERSION, body], sort_keys=True).encode("utf-8")).hexdigest()
    path = cfg.LLM_CACHE_DIR / key[:2] / f"{key}.json"          # a sub-folder per 2 hex digits
    stats["calls"] += 1
    if cfg.LLM_CACHE and path.exists():
        stats["cached"] += 1
        last_call.update(name=name, cached=True, seconds=0.0)
        return json.loads(path.read_text(encoding="utf-8"))["answer"]

    request = urllib.request.Request(f"{cfg.LLM_URL}/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as reply:
        out = json.load(reply)
    answer = json.loads(out["choices"][0]["message"]["content"])
    missing = [k for k in schema.get("required", []) if k not in answer]
    if missing:
        raise ValueError(f"{name}: reply lacks {missing}")
    seconds = round(time.perf_counter() - start, 1)
    shown = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else
              " ".join(p.get("text", "[image]") for p in m["content"])} for m in messages]
    if cfg.LLM_CACHE:
        atomic_write_json(path, {"name": name, "prompt_version": PROMPT_VERSION, "model": cfg.LLM_MODEL,
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

def llm_plan(state: AgentState) -> dict:
    # Ask the local model which tool to call next.
    # Returns {"tool", "query", "reason"}. Raises OSError if the server cannot be
    # reached, ValueError on a reply that is not the required JSON -- the caller
    # (planner.propose_action) decides what to do.
    goals = "\n".join(f"{i}. [{g['modality']}] {g['text']}" for i, g in enumerate(state["sub_goals"], 1))
    tried = ", ".join(f"sub-goal {g + 1} with {t}" for g, t in state["tried"]) or "nothing yet"
    missing = (state.get("sufficiency") or {}).get("missing") or "nothing checked yet"
    system = ("You plan retrieval for questions about one scientific paper. Pick ONE tool: "
              "'text' searches paragraphs, 'table' searches tables, 'figure' searches charts. "
              "Do not repeat a tool already tried for the same sub-goal. Give a one-sentence "
              "reason, then the tool, then a short search query.")
    user = (f"Question: {state['question']}\nSub-goals:\n{goals}\n"
            f"Already tried: {tried}\nStill missing: {missing}")
    schema = {"type": "object",
              "properties": {"reason": {"type": "string"},
                             "tool":   {"type": "string", "enum": list(cfg.MODALITIES)},
                             "query":  {"type": "string"}},
              "required": ["reason", "tool", "query"]}
    answer = call_llm([{"role": "system", "content": system}, {"role": "user", "content": user}], schema, "plan")
    return {"tool": answer["tool"], "query": answer["query"], "reason": answer["reason"],
            "seconds": last_call["seconds"]}
