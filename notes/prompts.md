# Prompt changelog

Every prompt the model sees, the reason for its wording, and what changed. Raise
`PROMPT_VERSION` in `src/gea/llm.py` with every change: the cache then stops reusing
replies to the old wording. The exact prompt text of every cached call is also saved in its
cache entry (`results/llm_cache/*/*.json`, field `messages`).

All calls: Qwen2.5-VL-7B-Instruct (Q4_K_M) through llama.cpp, temperature 0, seed 42, and a
JSON schema enforced by llama.cpp.

## v1 (2026-10-11)

### N1, splitting the question (`question.py`, `N1_SYSTEM`)

The answer is 1 to 3 sub-questions, each `{text, modality ∈ text | table | figure}`. The prompt
spells out what each kind of evidence usually holds:

- table: exact numbers comparing methods, datasets or settings;
- figure: trends, curves, diagrams, examples;
- text: definitions and explanations.

**Why:** in a first probe without that guidance, the model sent "Which method has the lowest
error on the test set?" to *text*, although such comparisons sit in tables (2026-10-11). The
prompt also says most questions stay one sub-question, so simple questions are not split
into pieces.

### N2, names (`question.py`, `N2_SYSTEM`)

The answer is up to 6 names the question is about: datasets, methods, metrics, variables,
written exactly as in the question. General words ("method", "dataset") are left out.

- Only names that occur in the question are kept (code check). A name the model invents can
  then never become a filter.
- Numbered references ("Table 2", "Section 4") stay regular expressions: they are read exactly
  that way and never wrongly.

### N5, the planner (`llm.py`, `llm_plan`)

Unchanged from the Day 1 notebook (Section 14): one tool from the schema's enum, a one-sentence
reason, and a short query. The model sees only what the rule planner sees: the sub-goals, the
tools already tried, and G1's "missing".

### G1, enough evidence? (`sufficiency.py`, `G1_SYSTEM`)

The answer is `{sufficient, uncovered: [sub-question numbers], missing}`.

- The question is "sufficient only if, for EVERY sub-question, some item contains the specific
  fact, number or picture that answers it -- not merely the same topic". This is meant
  against the lenient-judge failure.
- The judge sees each buffer item as text: paragraphs and tables in full up to 1,500
  characters, figures by caption plus "(image attached)". Figure images (the zoomed crop, or
  the region) are attached only when a sub-goal needs a figure, at most 2.
- **Rule floor (code, not prompt):** "sufficient" counts only if every table or figure
  sub-goal has an item of that kind in the buffer.

## v2 (2026-10-11): N1 gets examples

The N1 prompt now asks where the answer is most likely PRINTED. It defines "figure" as
anything shown as a picture: curves, architectures and their parts, visual examples. It also
gives six general examples, written for this prompt and not taken from the dev questions.

**Why:** with v1, N1 gave the reference's kind on only 6/20 dev questions (figures 1/11) and
chose "text" 12 times, for example "Which method performed the best on the GRID dataset?" →
text. **Result:** 8/20 (tables 6/9, figures 2/11); the regex stand-in gets 2/20
(`scripts/eval_n1.py` → `results/day5_n1.json`).

Many remaining "misses" are fair "text" choices, since the paper's prose also answers them
("What is the relationship between clicks and impressions?"). So no further tuning was done
on 20 dev questions: it would only fit them. The examples resemble common SPIQA question
types, so the honest number is the one on the test split.

## v3 (2026-10-11): G1 must name its evidence

The G1 prompt now asks, when it says sufficient, for each sub-question's evidence item
(E1, E2, ...) and the answering fact in a few words. Schema: `answers: [{sub_question,
evidence, fact}]`.

**Code check** (`sufficiency.backed`): "sufficient" holds only if every sub-question names an
item that is in the buffer and whose numbers include every number in the stated fact. Figures
are exempt, because their numbers are in the image.

**Why:** with v1/v2 the judge said SUFFICIENT for "Which method performed the best on the
GRID dataset?" after reading three paragraphs *about* the results, none of which said which
method won.

**Limit, seen in the traces:** the judge can still name a real but irrelevant fact. For "Which
model performs the best in terms of AUC on MovieLens-1M?" it accepted Table 4 with "AUC of
0.920 with a ripple set size of 32" (true, but about ripple-set sizes, not about which model is
best). The code can check that a fact exists; it cannot check that the fact answers the question.

## Not a prompt, but it changes what the tools see (2026-10-11)

The tools now search with the planner's query plus the words of the sub-goal it works on
(`retrieval.search_query`). The model planner writes short keyword queries ("AUC MovieLens-1M
dataset"), and SPIQA questions reuse the paper's own words (Day 3).
