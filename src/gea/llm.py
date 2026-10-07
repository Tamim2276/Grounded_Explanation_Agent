# The local language model: Qwen2.5-VL-7B served by llama.cpp's llama-server on
# the B580 (BUILD_PLAN.md Day 1 installs it, Day 5 wires it into the nodes).
#
# The notebook only talks to LLM_URL; llama.cpp decides how to use the GPU.
import json
import time
import urllib.request

from gea import config as cfg
from gea.state import AgentState


def llm_ready() -> bool:
    # True if a llama.cpp server answers at LLM_URL.
    try:
        with urllib.request.urlopen(f"{cfg.LLM_URL}/models", timeout=3) as reply:
            return reply.status == 200
    except OSError:                   # not running, refused, timed out
        return False


def llm_plan(state: AgentState) -> dict:
    # Ask the local model which tool to call next.
    # Returns {"tool", "query", "reason", "seconds"}. Raises OSError if the server
    # cannot be reached, ValueError on a reply that is not the required JSON --
    # the caller (planner.propose_action) decides what to do.
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
    body = {"model": cfg.LLM_MODEL, "temperature": 0, "seed": cfg.SEED,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "plan", "schema": schema}}}
    request = urllib.request.Request(f"{cfg.LLM_URL}/chat/completions",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as reply:
        content = json.load(reply)["choices"][0]["message"]["content"]
    answer = json.loads(content)
    return {"tool": answer["tool"], "query": answer["query"], "reason": answer["reason"],
            "seconds": round(time.perf_counter() - start, 1)}
