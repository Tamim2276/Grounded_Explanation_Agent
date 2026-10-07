# Recording runs, and what each situation costs (notebook Sections 11-12 and 15).
from gea import config as cfg
from gea.retrieval import TOOL_NODES
from gea.state import AgentState, new_state
from gea.stub_paper import FIGURE_QUERY, QUERY, TEXT_QUERY, THREE_PART_QUERY, UNANSWERABLE

# Nodes that are one call to the vision-language model in deployment.
MODEL_NODES = {"n1_query_intake", "n2_query_encoder", "n5_agentic_planner",
               "g1_sufficiency_check", "n9_grounded_generator", "g2_faithfulness_check"}


def record_run(graph, state: AgentState) -> list:
    # Run one query and write down which node ran in which step.
    #
    # invoke() only hands back the final state. stream(stream_mode="debug") runs
    # the same graph but reports events as they happen; a "task" event means "this
    # node is running in this step" -- that is all we keep.
    #
    # Returns (step, node) pairs, e.g. [(1, "n1_query_intake"), (2, "n2_query_encoder"), ...].
    # Pairs with the same step ran in parallel.
    timeline = []
    for event in graph.stream(state, stream_mode="debug"):
        if event["type"] == "task":
            timeline.append((event["step"], event["payload"]["name"]))
    return timeline


def record_writes(graph, state: AgentState) -> list:
    # Run one query and record what every node wrote into the state. Like
    # record_run, but with stream_mode="updates". Returns (node, update) pairs in order.
    writes = []
    for event in graph.stream(state, stream_mode="updates"):
        for node, update in event.items():
            writes.append((node, update or {}))
    return writes


def describe_write(node: str, update: dict) -> str:
    # One short, readable line for what a node wrote.
    u = update
    if node == "n1_query_intake":
        return "; ".join(f"{g['modality']}: {g['text'][:38]}" for g in u["sub_goals"])
    if node == "n2_query_encoder":
        f = u["query_filters"]
        return f"refs {f['refs'] or '-'}, must mention {f['entities'] or '-'}"
    if node == "n5_agentic_planner":
        return f"use {u['action']['tool']}"
    if node in TOOL_NODES.values() or node == "n7_region_zoom":
        return ", ".join(f"{e.tool}: {e.source}" for e in u["new_evidence"][-1:]) or "nothing"
    if node == "n8_evidence_buffer":
        return f"{len(u['evidence'])} item(s): " + ", ".join(f"{e.id} {e.kind}" for e in u["evidence"])
    if node == "g1_sufficiency_check":
        v = u["sufficiency"]
        return "sufficient" if v["sufficient"] else f"missing {v['missing']}"
    if node == "n9_grounded_generator":
        return f"{len(u['claims'])} claim(s), {sum(len(c['cites']) for c in u['claims'])} citation(s)"
    if node == "g2_faithfulness_check":
        n = len(u["faith_failures"])
        return "every citation matches the buffer" if n == 0 else f"{n} claim(s) rejected"
    if node == "n10b_bounding_boxes":
        return "; ".join(sorted({b["what"] for b in u["boxes"]}))
    if node == "n12_final_output":
        return u["answer"]
    return "wrote " + ", ".join(k for k in u if k != "events")


def run_scenario(graph, question, lie_once: bool = False, lie_always: bool = False) -> list:
    # Set up one situation, record the run, then put every setting back -- even if
    # the run crashes (cfg.override does the putting back).
    #   question    the question to ask
    #   lie_once    the generator misquotes on its first draft, then corrects itself
    #   lie_always  the generator misquotes on every draft
    # Returns the timeline from record_run.
    with cfg.override(TRACE=False, HALLUCINATE_ONCE=lie_once, HALLUCINATE_ALWAYS=lie_always):
        return record_run(graph, new_state(question))


def run_with_style(graph, style: str = "honest", lie_once: bool = False) -> AgentState:
    # Run the running example with the table claim written one of four ways
    # (notebook Section 15). Everything is put back afterwards, even after a crash.
    with cfg.override(TRACE=False, TABLE_CLAIM_STYLE=style, HALLUCINATE_ONCE=lie_once,
                      HALLUCINATE_ALWAYS=False):
        return graph.invoke(new_state(QUERY))


def profile(timeline: list) -> dict:
    # Turn a timeline from record_run into the counts for the cost table.
    ran = [node for _, node in timeline]
    return {
        "steps":       len({s for s, _ in timeline}),
        "node_runs":   len(timeline),
        "model_calls": sum(node in MODEL_NODES for node in ran),
        "rounds":      ran.count("n5_agentic_planner"),
        "zooms":       ran.count("n7_region_zoom"),
        "rewrites":    ran.count("rewrite_prep"),
        "dropped":     "drop_unsupported" in ran,
        "abstained":   "abstain" in ran,
    }


# The situations. A key that is left out (for example lie_once) means "no".
SCENARIOS = [
    {"name": "Text question, one round",          "question": TEXT_QUERY},
    {"name": "Figure question, with zoom",        "question": FIGURE_QUERY},
    {"name": "Running example, two rounds",       "question": QUERY},
    {"name": "Running example, one misquote",     "question": QUERY, "lie_once": True},
    {"name": "Running example, never corrected",  "question": QUERY, "lie_always": True},
    {"name": "Unanswerable, abstains",            "question": UNANSWERABLE},
    {"name": "Three sub-goals, never corrected",  "question": THREE_PART_QUERY, "lie_always": True},
]

# What each run measured when the stub graph was finished. If a change to the
# graph alters a cost, the notebook (11D) and the tests find out.
EXPECTED_COSTS = {                               # (steps, node runs, model calls)
    "Text question, one round":          (11, 13,  6),
    "Figure question, with zoom":        (12, 14,  6),
    "Running example, two rounds":       (16, 18,  8),
    "Running example, one misquote":     (19, 21, 10),
    "Running example, never corrected":  (23, 25, 12),
    "Unanswerable, abstains":            (20, 20,  8),
    "Three sub-goals, never corrected":  (29, 31, 14),
}


def predicted_steps(rounds: int, zooms: int, rewrites: int, dropped: bool, abstained: bool) -> int:
    # How many steps a question should take, worked out from the graph's shape.
    # Nothing here runs the graph -- see the table in notebook Section 11.
    retrieval = 2 + 4 * rounds + zooms + (rounds - 1)       # N1, N2, the rounds, replan_prep
    if abstained:
        return retrieval + 4                                # abstain, N10c, N11, N12
    return retrieval + 2 + 3 * rewrites + (1 if dropped else 0) + 3


# Scenario names as they appear in the thesis, in the order the thesis lists them.
PAPER_ROW_NAMES = {
    "Text question, one round":          "Text, 1 round",
    "Figure question, with zoom":        "Figure, zoom",
    "Running example, two rounds":       "Running example",
    "Running example, one misquote":     "Running ex., 1 misquote",
    "Running example, never corrected":  "Running ex., never corrected",
    "Unanswerable, abstains":            "Unanswerable",
    "Three sub-goals, never corrected":  "Worst case (3 sub-goals)",
}


def write_cost_table(rows: list, path) -> str:
    # Write the cost table as a LaTeX tabular, straight from measured rows. Only the
    # tabular is written: the caption stays in the thesis, the numbers stay here.
    # The thesis must load booktabs (it already does). Returns the LaTeX text.
    by_name = {r["scenario"]: r for r in rows}
    lines = [
        "% Generated by notebooks/langgraph_demo.ipynb (test 11I).",
        "% Do not edit by hand -- re-run the notebook instead.",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Scenario & $R$ & $Z$ & $W$ & Steps & Node runs & Model calls \\",
        r"\midrule",
    ]
    for notebook_name, paper_name in PAPER_ROW_NAMES.items():
        r = by_name[notebook_name]
        lines.append(f"{paper_name} & {r['rounds']} & {r['zooms']} & {r['rewrites']} & "
                     f"{r['steps']} & {r['node_runs']} & {r['model_calls']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
