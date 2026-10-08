# A guided tour of the package (BUILD_PLAN.md, Day 1, step 1.10)

This tour follows **one question** through the code, stop by stop, the way the graph runs it.
Open each linked file next to this page (the links jump to the line). It takes 45–60 minutes.
It also works during an outage: read it on your phone from GitHub, or print it.

Line numbers can move a little as the code changes; the function names will not.

---

## The map

| File | What lives there | Nodes | Made real on |
|---|---|---|---|
| [state.py](../src/gea/state.py) | the shared whiteboard (`AgentState`) and the evidence format (`Evidence`) | — | — |
| [question.py](../src/gea/question.py) | understanding the question | N1, N2 | Days 3, 5 |
| [planner.py](../src/gea/planner.py) | deciding which tool to use next | N5 | Day 5 |
| [retrieval.py](../src/gea/retrieval.py) | the tools and the evidence buffer | N6a, N6b, N6c, N7, N8 | Days 3–4 |
| [sufficiency.py](../src/gea/sufficiency.py) | "do I have enough?" and the retrieval loop | G1 | Day 5 |
| [generation.py](../src/gea/generation.py) | writing the answer and checking it | N9, G2 | Day 6 |
| [proofs.py](../src/gea/proofs.py) | the three proofs, the report, the answer | N10a–c, N11, N12 | Day 6 |
| [graph.py](../src/gea/graph.py) | how the nodes are wired together | — | Day 7 |
| [corpus.py](../src/gea/corpus.py) | reading the paper and building the indexes | N3, N4 | Days 2–3 |
| [config.py](../src/gea/config.py) | every constant and switch | — | — |

The supporting files are covered after the tour.

---

## The tour: one question, start to finish

> *"Does the reported accuracy gain in Section 4 hold for the low-resource subset, and which table reports it?"*

This is exactly what the graph prints for it (`cfg.TRACE = True`). Every stop below explains
one or two of these lines:

```text
  [N1   ] 2 sub-goal(s): (1) text: Does the reported accuracy gain in Section 4 hold for the low-resource subset; (2) table: Which table reports it
  [N2   ] query -> (768,), refs={'section': '4'}, must mention=['low-resource']
  [N5   ] round 1: text <- "Does the reported accuracy gain in Section 4 hold for the low-resource subset"
  [N3   ] 2 pages -> 5 chunks, 1 table(s), 1 figure(s)
  [N4   ] text index (7, 768), page index (2, 1024, 128)
  [N6a  ] c4 (page 2, score 3.42): "Table 3 compares our model with the baseline on ..."
  [N8   ] added E1=text/c4; buffer holds 1 item(s)
  [G1   ] INSUFFICIENT: missing table evidence for sub-goal 2; any mention of 'low-resource'
  [REPLN] round 2/3: back to the planner
  [N5   ] round 2: table <- "Which table reports it low-resource"  [G1: missing table evidence ...]
  [N6b  ] t3 (page 2, score 2.71): "Table 3: Accuracy (%) of the baseline and our mo..."
  [N8   ] added E2=table/t3; buffer holds 2 item(s); links E1-E2
  [G1   ] SUFFICIENT
  [N9   ] draft 1: 3 claim(s)
  [G2   ] all 3 claim(s) verified against the buffer
  [N10a ] 3 claim(s) linked to 6 source(s)
  [N10b ] 6 box(es) over 5 region(s)
  [N10c ] TextRetriever -> G1 reject -> TableRetriever -> G1 pass -> draft 1 -> G2 pass
  [N11  ] report written (19 lines)
  [N12  ] status=ok: So the gain does hold for the low-resource subset, but it is sma...
```

### Stop 0: the whiteboard — [state.py](../src/gea/state.py#L17)

Before anything runs, [`new_state`](../src/gea/state.py#L66) writes the question on a fresh
whiteboard (`AgentState`). Every list starts empty, and `round` starts at **1** because the
first search needs nobody's permission.

Each node **reads** the whiteboard and **returns only the keys it changed**; LangGraph copies
those onto the whiteboard. One key is special: `events` is
`Annotated[list, operator.add]`, which means "add to the list, don't replace it". Every node
adds one line there through [`step()`](../src/gea/trace.py#L10) (that line is also what you see
printed). At the end, N10c turns `events` into the trace.

Also look at [`Evidence`](../src/gea/state.py#L102): the **one** format every tool returns. It
always has `kind`, `page`, `content` (what the writer reads) and `bbox` (where it is on the
page). Tables also carry `cells`; figures carry `similarity`. Because everything is one format,
N8, N9 and G2 never need to know which tool found an item.

### Stop 1: N1 splits the question — [question.py:9](../src/gea/question.py#L9)

[`n1_query_intake`](../src/gea/question.py#L9) cuts the question at ", and" (or ";") into
**sub-goals**, and guesses the kind of evidence each needs from keywords: the second part says
"table", so it gets `modality: "table"`; the first gets the default, `"text"`.

*Why it matters:* the second part could not even be searched for before the first is found.
That is the "multi-hop" problem. *On Day 5* this regex becomes a model call, because real
questions rarely say the word "table".

### Stop 2: N2 reads the filters — [question.py:29](../src/gea/question.py#L29)

[`n2_query_encoder`](../src/gea/question.py#L29) finds references (`Section 4` → `{"section": "4"}`)
and names the evidence **must mention** (`low-resource`, from "… subset"). It also makes the
question's embedding; for now a `FakeTensor` that only has the real shape `(768,)`.

*On Day 3* the embedding becomes a real SPECTER2 vector.

### Stop 3: N5 picks a tool — [planner.py:40](../src/gea/planner.py#L40)

[`n5_agentic_planner`](../src/gea/planner.py#L40) works in **two stages**:

1. [`propose_action`](../src/gea/planner.py#L23) asks for a suggestion: from
   [`rule_based_action`](../src/gea/planner.py#L8) now, from the language model on Day 5.
   The rules take the first sub-goal that is not yet covered (at the start: sub-goal 1) and
   the tool for its modality (text).
2. N5 then **checks** the suggestion: the tool must exist, and the pair (sub-goal, tool) must
   not have been tried. Whoever proposed, these rules always hold.

It writes `action` (tool + search query) and adds `(0, "text")` to `tried`.

### Stop 4: the signpost — [retrieval.py:18](../src/gea/retrieval.py#L18)

[`route_tool`](../src/gea/retrieval.py#L18) is a **conditional edge**: it only reads
`action["tool"]` and returns `"text"`. [`TOOL_NODES`](../src/gea/retrieval.py#L14) maps that
name to the node `n6a_text_retriever`. A signpost may not write anything, only point.

### Stop 5: N6a searches the text — [retrieval.py:42](../src/gea/retrieval.py#L42)

[`n6a_text_retriever`](../src/gea/retrieval.py#L42) first calls
[`get_corpus`](../src/gea/corpus.py#L96) for this paper. The first time anyone asks, that runs
the offline graph: N3 reads the paper and N4 builds the indexes. **That is why the `[N3]` and
`[N4]` lines appear here, after N5**: the corpus is built when it is first needed, then kept.

Then [`text_retriever`](../src/gea/retrieval.py#L32) keeps only chunks from Section 4 (N2's
filter) and ranks them with [`score`](../src/gea/text.py#L28): words shared by the query and
the chunk, rare words counting more (IDF). Chunk `c4` wins with 3.42.

*On Day 3* `score` is replaced by SPECTER2 + FAISS; the node itself does not change.

### Stop 6: N8 files the evidence — [retrieval.py:178](../src/gea/retrieval.py#L178)

[`n8_evidence_buffer`](../src/gea/retrieval.py#L178) gives the new item an id (`E1`), adds it to
`evidence` (skipping anything already there) and empties the inbox `new_evidence`. Then
[`link_evidence`](../src/gea/retrieval.py#L162) links related items.

**This is the only node that writes `evidence`**, the folder the writer (N9) will read. The
notebook's test 10G checks that on real runs.

### Stop 7: G1 says "not enough" — [sufficiency.py:22](../src/gea/sufficiency.py#L22)

[`g1_sufficiency_check`](../src/gea/sufficiency.py#L22) asks, for every sub-goal, whether the
buffer covers it ([`covers`](../src/gea/sufficiency.py#L14)): sub-goal 2 needs a **table** and
there is none. It also checks that "low-resource" appears somewhere: it does not. So it writes
the verdict `INSUFFICIENT` with what is missing.

Then the red diamond, [`sufficiency_gate`](../src/gea/sufficiency.py#L43), reads the verdict:
not sufficient, and round 1 < `MAX_ROUNDS` (3), so it points to `retrieve_again`.
[`replan_prep`](../src/gea/sufficiency.py#L54) adds 1 to `round`. Without that node the loop
could never count, because a signpost cannot write.

### Stop 8: N5 again, with what G1 said — [planner.py:8](../src/gea/planner.py#L8)

Now `uncovered` is `[1]` (sub-goal 2), so the rules choose the **table** tool. They also add
the missing word to the query: `"Which table reports it low-resource"`. This is the loop doing
multi-hop: what G1 learned changes what N5 asks next.

### Stop 9: N6b finds the table — [retrieval.py:72](../src/gea/retrieval.py#L72)

[`table_retriever`](../src/gea/retrieval.py#L55) ranks the tables by caption + cell text and
returns Table 3 **with every cell and its box on the page** (`cells`). Keeping those boxes is
what lets N10b draw them later. A table read as flat text would answer the question but could
not prove it.

### Stop 10: N8 links the two — `E1-E2`

Now the buffer holds `E1` (the paragraph) and `E2` (the table). `link_evidence` sees that they
are on the same page **and** that the paragraph mentions "Table 3", the table's label. So the
writer will get them as one connected piece of evidence, not two loose ones.

### Stop 11: G1 says "enough"

Both sub-goals are covered and "low-resource" now appears (in the table's rows), so the
signpost points to the writer.

### Stop 12: N9 writes cited claims — [generation.py:110](../src/gea/generation.py#L110)

[`n9_grounded_generator`](../src/gea/generation.py#L110) calls
[`draft_claims`](../src/gea/generation.py#L75), which writes one or more claims per buffer item:

1. [`text_claims`](../src/gea/generation.py#L25) quotes the best sentence of the paragraph,
   **word for word**: *"Averaged over all test data, our model achieves an accuracy gain of
   3.9 points over the baseline."*
2. [`table_claims`](../src/gea/generation.py#L32) reports the row the question asks about
   (Low-resource: Baseline 71.3, Ours 73.0, Gain +1.7), citing each **cell**,
3. and adds the conclusion, citing two cells: *"So the gain does hold for the low-resource
   subset, but it is smaller: +1.7 points versus +3.9 overall."*

Every claim carries `cites`: which evidence, and the exact quote or cell.
*On Day 6* a model writes the claims, in the same format.

### Stop 13: G2 checks every claim — [generation.py:160](../src/gea/generation.py#L160)

[`check_claim`](../src/gea/generation.py#L136) runs **equality tests**, no judgement:

- the claim cites something, and the cited evidence is in the buffer;
- a quote appears word for word in that evidence;
- a cited cell really has that value;
- every number in the sentence appears in what its citations vouch for
  ([`cite_context`](../src/gea/generation.py#L129) adds the label, so the "3" of "Table 3" counts).

All 3 claims pass. [`faithfulness_gate`](../src/gea/generation.py#L181) then returns a **list**,
`["links", "boxes", "trace"]`: returning several routes starts all three proof nodes at once
(a *fan-out*). If a claim had failed, it would have returned `"rewrite"` instead
([`rewrite_prep`](../src/gea/generation.py#L192) counts the rewrites), and after 2 rewrites
`"exhausted"` ([`drop_unsupported`](../src/gea/generation.py#L200) withholds the claim).

### Stop 14: the three proofs, in parallel — [proofs.py](../src/gea/proofs.py)

- [`n10a_claim_links`](../src/gea/proofs.py#L32): each claim → its sources (3 claims, 6 sources).
- [`n10b_bounding_boxes`](../src/gea/proofs.py#L52): a box per citation: the paragraph, and each
  cited cell (6 boxes; the Low-resource/Gain cell is cited twice, so 5 different places).
- [`n10c_dag_trace`](../src/gea/proofs.py#L82): the path through the graph, built from `events`.

They run in the same step, which is why `events` needed its "add, don't replace" rule.

### Stop 15: the report and the answer — [proofs.py:95](../src/gea/proofs.py#L95)

[`n11_report_synthesis`](../src/gea/proofs.py#L95) puts the answer and its proofs into one
report; [`n12_final_output`](../src/gea/proofs.py#L135) picks the short answer the user reads
first: the claim marked as the conclusion.

---

## How the nodes are wired — [graph.py](../src/gea/graph.py)

Read [`build_grounded_agent_graph`](../src/gea/graph.py#L121) last. It is just the tour above
written as edges, built from three helpers:

- [`add_retrieval`](../src/gea/graph.py#L22): N1 → N2 → N5 → (signpost) → one tool → N8
- [`add_g1_loop`](../src/gea/graph.py#L44): N8 → G1 → (signpost) → writer / `replan_prep` → N5 / `abstain`
- [`add_generation`](../src/gea/graph.py#L58): N9 → G2, and `rewrite_prep` → N9

Notice what is **not** there: no edge from N1 to N5. N5 needs N1's sub-goals, but it reads them
from the whiteboard. An edge would mean "run next", and would start N5 before N2 had finished
(the notebook's test 6C shows that crash).

---

## The supporting files

| File | In one sentence |
|---|---|
| [config.py](../src/gea/config.py) | All constants and switches. Read them as `cfg.NAME`; change them for one block with [`cfg.override(...)`](../src/gea/config.py#L49), which always puts them back. |
| [trace.py](../src/gea/trace.py) | `trace()` prints one line per node; `step()` prints it **and** returns it as an `events` entry. |
| [text.py](../src/gea/text.py) | Small helpers: `terms` (words that matter), `sentences`, `table_text`, and the keyword `score`. |
| [stub_paper.py](../src/gea/stub_paper.py) | The fake two-page paper and the example questions. Its numbers are consistent: the All row is the 80/20 average of the two subsets. |
| [corpus.py](../src/gea/corpus.py) | N3 and N4 for the fake paper, and `get_corpus()`, the one place every node gets a paper from (Day 2 adds the real papers here). |
| [safeio.py](../src/gea/safeio.py) | Power-cut-safe files: write to a temporary name and rename; append results one line at a time. |
| [llm.py](../src/gea/llm.py) | Talking to the local model through llama.cpp; the planner's model call. |
| [profiling.py](../src/gea/profiling.py) | Recording runs and counting what they cost (the thesis's execution-profile table). |
| [viz.py](../src/gea/viz.py) | Drawing: `show()` for code listings, the graph, the trace figure, the proof on the page. |
| [device.py](../src/gea/device.py) | Finding the B580 (PyTorch calls it `xpu`). |

---

## Try it yourself (15 minutes, in a notebook cell)

Each experiment changes one setting for one run and shows what the graph does:

```python
from gea import config as cfg
from gea.graph import build_grounded_agent_graph
from gea.state import new_state
from gea.stub_paper import QUERY, FIGURE_QUERY, UNANSWERABLE

agent = build_grounded_agent_graph()

# 1. Only one round allowed: G1 rejects, and the budget is spent -> it abstains instead of guessing.
with cfg.override(MAX_ROUNDS=1):
    print(agent.invoke(new_state(QUERY))["answer"])

# 2. The first draft misquotes a number: watch G2 reject it and N9 rewrite it.
with cfg.override(HALLUCINATE_ONCE=True):
    agent.invoke(new_state(QUERY))

# 3. A question the paper cannot answer: three rounds, then "I could not confirm".
agent.invoke(new_state(UNANSWERABLE))

# 4. A figure question: the figure is small on its page, so N7 zooms.
agent.invoke(new_state(FIGURE_QUERY))

# 5. G2's blind spot: a real number from the wrong row passes every equality test.
with cfg.override(TABLE_CLAIM_STYLE="misattributed"):
    print(agent.invoke(new_state(QUERY))["answer"])
```

For each one, find in the trace **which stop** behaved differently, and why.

---

## Check yourself

<details><summary>1. Why do the N3 and N4 lines appear after N5 in the trace?</summary>

N3 and N4 run inside `get_corpus`, and the first node that asks for the corpus is N6a. The
corpus is built then, once, and kept for every later question about the same paper.
</details>

<details><summary>2. What exactly makes the second round search for a table instead of text again?</summary>

G1 wrote `uncovered: [1]` (sub-goal 2 is not covered). `rule_based_action` works on the first
uncovered sub-goal, and that sub-goal's modality is "table".
</details>

<details><summary>3. Which node is the only one allowed to write `evidence`, and why does that matter for G2?</summary>

N8. Because nothing else can add to the folder, the writer's only possible source is that
folder, so G2 can check every claim against one fixed list instead of "the whole world".
</details>

<details><summary>4. Why is `faithfulness_gate` allowed to return a list?</summary>

Returning several route names starts all of those nodes in the same step. That is how the three
proofs run in parallel.
</details>

<details><summary>5. In experiment 5, why does G2 let the false claim through?</summary>

The claim cites the High-resource row's Gain cell, which really says +4.4, so every equality
test passes. Only a check that reads *which row the sentence is about* could catch it. Day 6
adds a model check for exactly this.
</details>
