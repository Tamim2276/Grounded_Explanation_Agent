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

## v4–v6 (2026-10-12): the writer (N9) and G2's support check

**N9** (`generation.N9_SYSTEM`): answer only from the evidence items; 1–4 claims; every claim
cites quotes, cells or (figures only) observations; never a number that is not in a cited quote
or cell; never read exact numbers off images; mark the conclusion. The JSON schema is built per
call: `evidence` is an enum of the buffer's ids, cells only for items with cells, observations
only for figures. On a rewrite, verified claims are kept and the failed ones are listed with
G2's reasons.

- **v4:** first real version. 39/51 claims passed both G2 layers (10 dev questions).
- **v5:** added "back claims about listed values with cells (for a trend, the first and last
  values)". Worse: 24/38. The model then quoted the formatted values lines, which are not
  paper text.
- **v6 (kept):** v5's sentence removed; "the 'Values (cite as cells)' lines are not text of the
  paper: cite them as cells, never quote them"; the row-label cells are dropped from the list.
  34/52. Not tuned further: the differences are within noise for 10 questions.

**G2 support check** (`generation.SUPPORT_SYSTEM`, layer 2): "supported only if the cited
evidence says it AS WRITTEN: the same subject, row, column, setting and number; a correct value
attached to the wrong subject or row is NOT supported". One call per draft over every claim that
passed layer 1, with the cited cells shown by row and column name, the cited items in full, and
figure images. A missing verdict counts as unsupported. On the stub boundary cases it rejected the
misattributed claim and nothing else (`results/day6_g2_boundary.json`).

## v7 (2026-10-12): G1 must state the answer itself; dev error analysis (Day 7.1)

Made after the error analysis of the 20 dev questions (`results/runs/full_dev_v6.jsonl`,
`notes/results_for_paper.md` §4e). G1 had said "sufficient" with a fact that was on the topic but
did not answer: "DA plays an important role" for "what is the *effect* of adding DA", an AUC by
ripple-set size for "*which model* performs best".

**G1** (`sufficiency.G1_SYSTEM`): added "The fact must BE the answer: for 'which X' it names the X;
for 'how does A affect B' or 'what is the effect of A' it states the direction or size of the
change; for 'how many' or 'what value' it gives the number. A fact that only restates the question
or describes the topic does not answer it." The **schema order** changed too: `answers`,
`uncovered`, `missing`, then `sufficient` last. llama.cpp fills the fields in schema order, so
before this the verdict was written first and the facts only justified it afterwards.

Every other prompt is unchanged; the version is global, so every reply is new under v7.

**Not a prompt, but it changes G2 layer 1:** a quote is compared with its evidence after undoing
PDF text-extraction quirks on both sides (`text.loose`): ligatures, curly quotes, line-end
hyphenation, spaces before punctuation, spaced decimals. Four correct claims had been withheld for
these (e.g. "in- ference", "0 . 5%"). A changed word or number still fails. And a rewrite no
longer repeats a kept claim (the same claim appeared up to 3 times).

**Gemini (comparison system):** the same prompts and schemas, sent to `gemini-3.8-flash` through
Google's OpenAI-compatible endpoint. `seed` is not accepted there (HTTP 400), so it is left out
for Gemini only; temperature is 0 and every reply is cached as for the local model.

**Length limits (still v7):** the first v7 run showed G1 repeating itself inside `missing` for
5,800 tokens (88 s) until the reply broke off as invalid JSON, and G1 then fell back to its rules.
So `fact` and `missing` have `maxLength: 300` (llama.cpp enforces it token by token), and every
request has `max_tokens: 2048` (`cfg.LLM_MAX_TOKENS`; the longest real reply on dev was about 460).
The v7 numbers are from the run with these limits.

## v8 (2026-10-12): tried and dropped

v7 refused 13 of 20 dev questions, including the GRID table, because it did not "explicitly
state" the best method. v8 added "The answer may need a simple reading of the evidence: comparing
the values in a table or chart, or a description in the text", named 'what is the role of X'
questions, and made `evidence` an enum of the buffer's ids (v7 once wrote "E1, E2, E5, E6"). On
dev: 5 right, 2 partly, **3 wrong**, 10 none (v7: 6, 1, **0**, 13). **v7 restored**, and tuning
stopped (`notes/error_analysis_dev.md`).
