# Results for the thesis

Every number the build has produced, grouped by the thesis chapter it belongs to. Each
one names its **source** (a file in the repository, or the command that made it), so it
can be checked and regenerated. The journal (`notes/journal.md`) keeps the same facts in
date order, together with what broke and how it was fixed.

**Dev or test?** Numbers on the **dev** split (20 questions) are development results. They
guided design choices, so the thesis should call them *pilot* or *development* results. The
final Results chapter uses the **test** split (150 questions), which is run once on Days 7–8.

---

## 1. Experimental setup → Methodology, *Experimental Setup*; Implementation

### Hardware and software

| item | value | source |
|---|---|---|
| GPU | Intel Arc B580, 12 GB (12,190 MiB visible to Vulkan) | llama.cpp start log, 2026-10-08 |
| RAM | 15.6 GB, plus a 15 GB page file | Windows, 2026-10-09 |
| Python / PyTorch | 3.12.12 / 2.14.1+xpu | `.venv` |
| Agent framework | LangGraph 1.2.14 | `.venv` |
| PDF reading | PyMuPDF 1.28.2, pages rendered at 150 dpi | `src/gea/ingest.py` |
| Text embedder | SPECTER2: `allenai/specter2_base` + `proximity` adapter (paragraphs) + `adhoc_query` adapter (questions), 768 dimensions, float32 | `src/gea/indexes.py` |
| Text index | FAISS 1.15.1 `IndexFlatIP` (exact inner product = cosine on length-1 vectors) | `src/gea/indexes.py` |
| Page embedder | ColQwen2: `vidore/colqwen2-base` + `vidore/colqwen2-v1.0` LoRA, merged; bfloat16; 128 dimensions per patch | `src/gea/indexes.py` |
| Libraries pinned together | transformers 4.47.1, adapters 1.1.0, colpali-engine 0.3.8 | `requirements.txt` |
| Language model (from Day 5) | Qwen2.5-VL-7B-Instruct, Q4_K_M GGUF + f16 vision projector, llama.cpp build 11476 (commit 988190680), Vulkan | `scripts/start_llm.sh` |
| Operation | fully offline: every model loads from the local download only | `local_snapshot()` in `src/gea/indexes.py` |

### Data → Methodology, *Dataset*

| item | value | source |
|---|---|---|
| Source | SPIQA test-A: 118 papers, 666 questions | `scripts/download_all.py` |
| Question selection | seed 42; **whole papers per split** (no paper in both); at most 4 (dev) / 8 (test) questions per paper | `scripts/select_questions.py`, `eval/splits/summary.json` |
| Dev split | 20 questions from 5 papers: 9 table, 11 figure | `eval/splits/summary.json` |
| Test split | 150 questions from 31 papers: 71 table, 79 figure | `eval/splits/summary.json` |
| Roman-numeral labels ("Table II") | 7 of the 666 references; 1 in test | journal, Day 2 |
| Annotated subset (CGS) | 50 questions, still to annotate (Day 6–7) | plan |

---

## 2. Ingestion (N3) → Implementation, *Data and Ingestion* (Day 2)

| item | value | source |
|---|---|---|
| Papers read | 36 (all dev and test papers), 475 pages | `results/day2_coverage.json` |
| Records | 2,168 text chunks (≤ 150 words, never crossing a page, section, column or table), 159 tables, 205 figures | same |
| **Coverage**: the answer's table/figure was found | **dev 20/20, test 150/150 (100%)**; target was 85% | same |
| How the 364 regions were found | 63 PyMuPDF table finder (then grown), 300 caption strip, 1 last resort | same |
| Regions marked doubtful | 3 of 364 | `corpus.json` files |
| Time | about 90 s on the CPU for all 36 papers | `scripts/build_corpus.py` output |

**Problems found on real papers, each fixed and kept by a test** (`tests/test_ingest.py`). Good
material for the Implementation chapter; the details are in the journal:

1. A caption glued to the table rows under it in one PyMuPDF block. Fixed by finding captions per row and stopping at the first row whose pieces are more than one font size apart.
2. Two regions claiming the same content (a table stacked on a figure). Fixed by placing tables first and treating them as walls.
3. PyMuPDF's table finder missing or splitting tables. Its guess is used only next to a table caption, then grown over neighbouring table text.
4. Papers that put table captions *under* their tables. Fixed by a per-paper vote over two passes.
5. Table rows taken for paragraphs. Fixed by a "table-like" test: mostly gapped rows, or at least 40% numbers.
6. A page number taken for a table. Fixed by ignoring the top and bottom 40 pt margins.
7. IEEE captions ("TABLE I", with the title on the next line).
8. Words broken at a line end ("be- tween").

**Known limitations:**
- "state-of-the-" + "art" becomes "state-of-theart".
- 1706.00633v4 draws Table 2 and Figure 2 in one frame, so the Figure 2 region is doubtful.
- One table's ruling lines are invisible to PyMuPDF; it was found by the last-resort search.

---

## 3. Indexing (N4) → Implementation, *Indexing* (Day 3)

| item | value | source |
|---|---|---|
| Text index rows | 2,532 = 2,168 chunks + 364 captions | `scripts/build_indexes.py text` |
| Text index build | **0.9 min on the B580**; the CPU managed ~4 chunks/s while another job ran (~10 min) | build output, 2026-10-09 |
| Page index | 475 pages; a 150-dpi page (1275×1650 px) is resized to 672×868, giving **31 × 24 = 744 patch vectors** of 128 numbers | `scripts/build_indexes.py pages` |
| Patch size on the page | about 25.5 pt (9 mm) square on a 612×792 pt page | computed |
| Page index build | **3.0 min on the B580** (~0.3 s a page); planned 1–2 h | build output, 2026-10-09 |
| Page index size | ~1.6 MB per paper (bfloat16) | `data/corpus/*/pages.pt` |
| Model load time | SPECTER2 ~4 s; ColQwen2 ~7 s (two weight shards) | logs |

---

## 4. Retrieval → Results, *RQ2* (Day 3, dev, pilot)

Source: `results/day3_retrieval.json`, from `python scripts/eval_retrieval.py` (20 s on the B580).
The reference is the table or figure each question's answer lives in.

| measure (20 dev questions) | measured | random guess |
|---|---|---|
| page hit@1: ColQwen2 ranks the reference's page first | **75%** (15/20) | 12% |
| page hit@3 | **95%** (19/20) | 37% |
| mean reciprocal rank of the reference's page | **0.847** | |
| text hit@3, lenient: one of SPECTER2's top 3 rows is the caption, names the label, or is on that page | 75% (15/20) | |
| text hit@3, **strict**: the caption, or a paragraph naming the label ("Table 2", "Fig. 2") | **50%** (10/20) | |
| page hit@3 **or** strict text hit@3 | **100%** (20/20) | |

**Report the strict text number.** The 15 lenient hits split evenly: 5 were the reference's own
caption, 5 were paragraphs naming it, and 5 were *only on the same page*, which can be
coincidence. In 1803.03467v4#0, for example, the page-6 paragraph is about semantic matching
models, not the dataset table. (Classified from `top3_text` in `results/day3_retrieval.json`,
2026-10-09. From now on `scripts/eval_retrieval.py` records `text_match` and
`text_hit@3_strict` itself.)

**The measure can also undercount.** For 1804.07931v2#1 ("What is the relationship between
clicks and impressions?", reference Figure 1 on page 1), SPECTER2's top 3 included the page-2
sentence "S_c is a subset of S" (clicked impressions are a subset of all impressions), which
states the answer. It counts as a miss because it is neither the figure nor on its page.
Retrieval here is measured against the *reference figure*, not against whether the answer is
stated somewhere.

| by kind | page hit@1 | page hit@3 | text hit@3 (lenient) |
|---|---|---|---|
| tables (9) | 8/9 | 9/9 | 8/9 |
| figures (11) | 7/11 | 10/11 | 7/11 |

**What to say about it:**
- Looking at page *images* finds the right page far above chance. That is first evidence for RQ2.
- **Page images beat text for finding tables and figures:** 95% page hit@3, against a strict 50% for text.
- **The two indexes complement each other.** The one page outside the top 3 (1708.00160v2, Figure 2) was found by text: a paragraph naming "Figure 2". So every dev question was within reach of at least one index, which supports the dual-index design (N4).
- Figures are harder than tables.
- The only page outside the top 3 was 1708.00160v2, Figure 2 (rank 10 of 15). Worth an error-analysis look on Day 4.
- Margins can be thin. For 1803.03467v4#0 (Table 1, page 6), page 6 scored 13.48 and page 7 scored 13.27.

### Text search for N6a: keyword vs SPECTER2 vs hybrid (Day 3, dev, pilot)

Source: `results/day3_text_search.json`, from `python scripts/compare_text_search.py` (CPU, 5 s,
2026-10-09). This is the real N6a code, returning the top **3 paragraphs**; caption rows are
excluded. A hit means a returned paragraph is about the question's reference: **strict** means
it names the label ("Table 2", "Fig. 2"); **lenient** also counts a paragraph on the same page.

| method | strict@1 | strict@3 | lenient@3 |
|---|---|---|---|
| keyword (IDF-weighted word overlap; ties in paper order, no model) | 45% | **70%** | **90%** |
| SPECTER2 (question adapter + FAISS) | 15% | 35% | 70% |
| hybrid (reciprocal rank fusion, k = 60) | **50%** | 60% | 80% |

**What to say about it:**
- **Within one paper, exact-word matching beats the scientific text embedder for finding paragraphs.** SPIQA questions reuse the paper's own terms: "GRID", "DA", "ripple sets", "CVR and CTCVR", "{head=F, ant=NAM}". SPECTER2 encodes topic, and every paragraph of a paper shares the topic. SPECTER2 alone won on only one question (1803.03467v4#2, "AUC on MovieLens-1M").
- **Choice made: hybrid** (`cfg.TEXT_SEARCH`).
  - It is best at putting a correct paragraph first.
  - It is 2 questions of 20 behind keyword at top 3 (1812.06589v2#3, 1708.00160v2#0), within noise for n = 20.
  - It keeps a meaning signal for queries that do not reuse the paper's words. From Day 5 the planner writes sub-queries in its own words, and on Day 3.2's "few labelled examples" vs "low-resource", keyword scores 0.
- **To revisit:** re-run this comparison with the planner's own sub-queries on Day 5, and report all three on the test split as a retrieval ablation (Days 7–8; it costs seconds).

### SPECTER2: topics, not answers (Day 3, hand-made examples)

Question: "How well does the method work when only a few labelled examples are available?"

| paragraph | words shared with the question | SPECTER2 cosine |
|---|---|---|
| A, answers it ("low-resource setting … 71.3% accuracy") | none | 0.769 |
| B, only shares words ("labelled examples were collected by three annotators") | 2 | **0.779** |
| C, unrelated ("20 epochs on four GPUs") | none | 0.733 |

**A real case** (dev 1804.07931v2#1, "What is the relationship between clicks and impressions?",
answer: clicks are a subset of impressions; 2026-10-09). The answer-stating chunk c13 ("M is
the number of clicks over all impressions. Obviously, S_c is a subset of S") ranks:

- **1st by keyword search**, with IDF score 4.07; the next chunk scores 1.21;
- **2nd by SPECTER2**, at 0.783, behind c11 (0.797), a paragraph about CVR modelling on the same topic.

Both have it in the top 3, so keep more than one paragraph and consider mixing the two scores.

Over 4 such questions, the unrelated paragraph always came last, but the answer beat the
word-sharer only 2 times out of 4. All the scores fall between 0.65 and 0.78. SPECTER2 was
trained to match a search with papers *on the same topic*, so it ranks topic, not
answerhood. Inside one paper almost everything is on-topic. This motivates the top-3
retrieval, the page index as a second opinion, and G1's sufficiency check. Keyword
matching scores A and C the same (both 0).

---

## 4b. Table and figure tools (N6b, N6c, N7) → Implementation, *Retrieval tools*; Results, *RQ2* (Day 4, dev, pilot)

### Table structure (Table Transformer)

| item | value | source |
|---|---|---|
| Tables parsed | 159 in **0.9 min on the CPU**: 157 by the Table Transformer; 2 without a grid (one is a boxed text example, the other a table TATR saw as one column), for which the raw region text is kept | `python scripts/build_indexes.py tables`, `data/corpus/*/tables/` |
| Median table size | 7 rows × 5 columns | same |
| Cell text | read from the PDF's text layer (no OCR): the words whose **centre** lies in the cell | `src/gea/tables.py` |
| Checked by eye | 3 dev tables, every cell matches the PDF (incl. two-level headers) | `data/debug/day4/*_cells.png` |
| Example | 1803.03467v4 Table 1: "# 4-hop triples / Bing-News = 6,322,548", box (510.2, 172.7, 555.6, 183.0) pt | `tests/test_tables_figures.py` |

**Two problems on real tables, fixed:**

1. Reading every word that *touches* a cell box also took the next row's words, because the Table Transformer's row boxes overlap a little. Fixed by assigning each word to the cell holding its centre.
2. Words on slightly different baselines came out of order ("et al. [2017] Chung"). Fixed by grouping words into lines within 3 pt.

### Choosing the right table or figure

Source: `results/day4_regions.json`, from `python scripts/eval_regions.py` (2026-10-11, with
ColQwen2's query vectors; about 10 s once they are cached). The reference's kind decides the
tool (table → N6b, figure → N6c). A hit means the reference is ranked 1st (hit@1) or in the top
3 among all tables or figures of its paper.

| how the table/figure is chosen | tables (9): hit@1 / @3 | figures (11) | all (20) |
|---|---|---|---|
| **text**: caption (SPECTER2) + caption and the words inside the region (keyword) | 89% / 100% | 73% / 82% | 80% / 90% |
| **visual**: ColQwen2's MaxSim score of the region's *page* (ties by text) | **100%** / 100% | 64% / 91% | 80% / 95% |
| **both** (rank fusion of the two; `cfg.REGION_SEARCH`, the default) | 89% / 100% | **82%** / 91% | **85% / 95%** |
| heat: ColQwen2's heat *inside* the region (mean of its 5 hottest patches) | 56% / 100% | 55% / 82% | 55% / 90% |
| text + heat | 78% / 100% | 73% / 82% | 75% / 90% |
| random | 35% | 19% | 26% |

**Ablation, captions only** (no words from inside the region): tables 6/9 first, figures 8/11.
**Adding the region's words lifted tables to 8/9.** For example, "Which dataset has the most
4-hop triples?" matches Table 1's *cells*, while its caption only says "Basic statistics of the
three datasets."

### What ColQwen2's heat map can and cannot do (RQ2, RQ3)

| measure (dev, 20 questions) | value | random |
|---|---|---|
| pointing: the hottest patch of the reference's page lies inside the reference | 15% (tables 22%, figures 9%) | 8% (the reference's share of its page) |
| IoU of the box around the page's hottest patches with the reference | median 0.07 | |
| IoU of N10b's box (hot patches kept to the chosen region, clipped to it) with the whole region | median **0.81** (tables 0.79, figures 0.84) | |

**What to say about it:**
- **ColQwen2 is strong at *which page*** (Day 3: right page 1st 75%, top 3 95%; here its page score alone picks the right table 9/9). That is what it was trained for.
- **Its patch-level heat map is noisy.**
  - The words of the question light up wherever they appear: in the figure, in its caption, and in the paragraph that discusses it. Blank margins light up too.
  - Example: `data/debug/day4/figure4_heat.png`, and the per-word best patches on the Table 1 page, `data/debug/day4/heat_tokens_p6.png`. Individual words ("triples", "4") do land on the right table rows.
  - So the heat inside a region is a poor way to *choose* between regions (55% vs 85%). It is used only to narrow the box inside the chosen region.
- **Tried, no better** (dev pointing / heat choice):
  - max over content words only: 10% / 50%;
  - mean over content words: 10% / 60%;
  - z-scored mean: 10% / 45%;
  - the current max over all query tokens: 15% / 55%.
- **Orientation is right.** With the grid read row by row, the reference is hotter than the rest of its page by +0.035 on average (hotter in 70% of questions). With rows and columns swapped, that drops to +0.016, and pointing falls from 15% to 10%. This matches the code reading of Qwen2-VL's patch order.
- **For RQ3:** the heat-map box is a *refinement inside* a region chosen by other signals. Whether it beats the whole region as a box is a CGS question for the annotated subset (Days 6–8). Report both.

**Bug fixed on the way (also affects N6a):** in hybrid rank fusion, items that a search did not
find at all (keyword score 0) tied for a rank and still earned credit. With 5 tables, that let
SPECTER2's slight preference for Table 5 beat the only keyword match (Table 1). Now a score of 0
earns nothing, as in standard reciprocal rank fusion. Day 3's text numbers were re-run and are
unchanged.

### Zoom (N7) and the whole graph

- **Zoom:** a figure under 30% of its page (on the *real* page size) is re-drawn from the PDF at up to 300 dpi, long side ≤ 1024 px. For example, Figure 4 of 1803.03467v4 covers 9% of the page and becomes a 1025 × 770 px crop with readable axis labels (`data/debug/day4/zoom_1803.03467v4_Figure4.png`).
- **Smoke test** (`python scripts/smoke_real.py` → `results/day4_smoke.json`):
  - 5 dev questions ran through the **whole graph** on real papers with the rules planner.
  - 4 of them had N1's modality set from the reference kind, a testing knob, because N1's regex stand-in cannot tell a table question from a text one.
  - All 5 ran end to end. With ColQwen2 on the free B580 (2026-10-11) they took 0.0–26 s; the 26 s is loading ColQwen2 once. The reference table or figure reached the evidence buffer in 4 of 5.
  - Figure boxes now come from the heat map kept to the figure ("patch map"); table boxes are the region or the cited cell.
  - The answers are template sentences until the language model arrives (Days 5–6).

### Heat-map orientation

The saved patch vectors are read as a rows × columns grid, row by row. Two checks agree:

- **The code:** Qwen2-VL's image processor orders tokens as (row block, column block).
- **The data:** with this layout the reference is hotter than the rest of its page by +0.035; swapped, by only +0.016 (see above).

A model-free check, predicting each patch's ink from its vector, was too weak to decide
(correlation 0.12 vs 0.08): ColQwen2's patch vectors encode meaning, not ink.

---

## 4c. The language model in the loop (N1, N2, N5, G1) → Implementation, *The language model*; Discussion (Day 5, dev, pilot)

### The model on this PC

| item | value | source |
|---|---|---|
| Model | Qwen2.5-VL-7B-Instruct, Q4_K_M GGUF + f16 vision projector, llama.cpp build 11476, Vulkan, on the B580 | `scripts/start_llm.sh` |
| Settings | temperature 0, seed 42, JSON schema enforced by llama.cpp, `--image-min-tokens 1024` (llama.cpp's advice for Qwen-VL), context 8192 | `src/gea/llm.py` |
| GPU memory | ~6.8 GB (the GPU at ~7.6 GB in all) | Windows counters, 2026-10-11 |
| **Memory reserved on Windows** | **~7.5 GB** of commit (RAM + page file), only ~4.3 GB in RAM, whatever the flags (`--no-host` and `--load-mode mmap` made no difference) | Task Manager / PowerShell |
| Speed | ready in ~10 s; generation 55–63 tokens/s; prompt reading 90–175 tokens/s; a short JSON answer ~3–4 s, with one page image ~6–9 s | llama-server log |
| All model calls valid JSON | 50 + 42 calls (first two runs), 0 invalid, 0 fallbacks while the server ran | `results/day5_traces.json` |

**The shared-PC constraint, for Implementation/Limitations.**
- The model server's 7.5 GB reservation and ColQwen2's ~5 GB do not both fit beside other jobs on a 15.6 GB machine.
- So ColQwen2 runs *before* the model server, embedding the questions (cached in `data/cache/page_queries/`). At question time the page score comes from those cached question vectors (`retrieval.page_signals`).
- `scripts/start_llm.sh` refuses to start with less than 10 GB free commit.

### N1: which kind of evidence? (`scripts/eval_n1.py` → `results/day5_n1.json`)

| N1 version | gives a sub-goal the reference's kind (20 dev) | tables | figures |
|---|---|---|---|
| regex stand-in | 2/20 | | |
| model, prompt v1 | 6/20 | 5/9 | 1/11 |
| model, prompt v2 (with examples) | 8/20 | 6/9 | 2/11 |
| always "figure" (the majority kind) | 11/20 | 0/9 | 11/11 |
| **vote of the k nearest labelled questions** (SPECTER2 question vectors; 469 questions from the 82 SPIQA test-A papers *not* used for dev/test: 180 table, 289 figure); k = 5 / 9 / 15 | 16 / **17** / 17 of 20 | 5 / 6 / 6 of 9 | 11/11 |

**Adopted (2026-10-11): N1's kind of evidence comes from this vote**, with k = 9 fixed before
the test split is run (`src/gea/kinds.py`, `cfg.KIND_ROUTER = "vote"`, real papers only). The
model or the rules still split the question into sub-questions. The vote learns from labelled
questions on held-out papers, so there is no dev/test leakage:
`tests/test_kinds.py` checks that none of the 469 examples is a dev or test question.
`python scripts/eval_n1.py` reproduces the 17/20 (`results/day5_n1.json`, key `vote`).

**Caveats:**
- It can only answer table or figure, never text, because SPIQA labels every reference as one
  of these. It is a *learned router* for questions about a paper's tables and figures.
- 20 dev questions is a small sample; the honest number is the test split's.

**Design note, for the Method/Discussion chapters:** a 7B model asked "where would the answer
be printed?" defaults to "text". The same decision learned from ~470 labelled questions is
right 17/20 times. The language model is used where it is good (splitting, judging with
named evidence), and simple supervised data where it is not.

- The model mostly answers "text" (11 of 20 with v2).
- Many "misses" are fair: the paper's prose also answers "What is the role of the ripple sets?" or "What is the relationship between clicks and impressions?". The measure is rough because SPIQA labels every question with a figure or table.
- No further tuning on 20 dev questions. The prompt history is in `notes/prompts.md`.

### Ten dev questions with the model in charge (`scripts/run_traces.py`, prompt v3)

- **Runs:** 10 questions, 42 model calls, 0 invalid replies, 147 s on the first run (12–21 s per question).
- **Rerun from the cache:** **6.9 s**, identical.
- **Power-cut test:** with the server stopped, all 10 replayed **identically** from the cache. A new question fell back to the rules on N1, N2 and G1, and its trace said so.
- **Outcome, before and after the vote:**

  | 10 dev questions (prompt v3, model in N1, N2, N5, G1) | N1 gives the right kind | right table/figure reaches the buffer | abstained |
  |---|---|---|---|
  | kinds from the model | 4/10 | 3/10 | 0 |
  | **kinds from the vote** (`cfg.KIND_ROUTER = "vote"`) | **9/10** | **7/10** | 0 |

  With the vote, 50 model calls were made (30 from the cache) in 89 s, with 0 fallbacks
  (`results/day5_traces.json`). The answers are still template sentences of the stand-in
  generator; the model writes answers from Day 6.
- With the model's kinds, the planner's first tool matched N1's kind in 9/10.
- **G1's rule floor at work** (1812.06589v2#2, "Which method performed the best on the GRID dataset?", gold "AMIE (Ours)", Table 2):
  - N1 said table, but the planner tried text first.
  - G1 → INSUFFICIENT: "missing table evidence for sub-goal 1 (rule floor)".
  - Round 2: table → Table 2.
  - G1 named "E4: AMIE (Ours) performed the best on the GRID dataset", the gold answer. Replayed in notebook Section 20.
- **The lenient judge, before and after v3:**
  - With v1/v2, G1 accepted three paragraphs *about* the GRID results as sufficient.
  - v3 makes the judge name the item and the fact. Code checks the item exists and the fact's numbers are in it.
  - **Remaining failure:** a true but irrelevant fact still passes. For "which model performs best in AUC on MovieLens-1M?" it accepted Table 4's "AUC 0.920 with ripple set size 32" (Table 4 is about ripple-set sizes; the answer is in Table 3). Day 8 measures both failure modes: wrong answers and abstentions.

### Reading charts: a limitation of the 7B model (2026-10-11)

On Figure 4 of 1803.03467v4 (four bar-chart panels; zoomed crop, 1025 × 770 px), asked to read one bar:
- panel (c) at hop 4, true ≈ 55: answered **12**;
- panel (a) at hop 1, true ≈ 4: answered **25**, which is the hop-2 value.

Both were wrong, with and without `--image-min-tokens 1024`. So **figure answers that depend on reading exact values are unreliable**; Day 6 should prefer values written in text or tables. The G2 faithfulness check cannot verify numbers read from an image. This is an honest limitation for RQ1/RQ3.

**A fix for vector charts: read the values from the PDF itself** (exploratory prototype, 2026-10-11;
it becomes package code with tests on Day 6).

Many CS papers embed plots as *vector graphics*: every bar is an exact rectangle and every axis
number is PDF text. Calibrating each panel's y-axis from its tick labels (a straight-line fit of
value against height), then converting each bar's top, gives:

| bar (Figure 4, 1803.03467v4) | 7B model from the image | read from the PDF |
|---|---|---|
| (c) Bing-News, hop 4, with common raters (true ≈ 55) | 12 | **55.8** |
| (a) MovieLens-1M, hop 1, with common raters (true ≈ 4) | 25 | **4.3** |

All 22 bars of panels (a)–(c) were read correctly, each with its exact rectangle as a box. Panel (d)
is a line chart, which the bar reader does not cover.

**How many figures this helps** (205 figures of the 36 papers, a rough heuristic classification):

| kind | share | what can be checked exactly |
|---|---|---|
| vector diagrams (architecture, pipeline, tree) | 94 (46%) | their text labels (PDF text) |
| raster pictures (photos, screenshots, pasted plots) | 67 (33%) | nothing; only a vision model can read them |
| vector charts with axes and lines/points | 34 (17%) | values, with a line-chart reader (more work) |
| vector bar charts | 10 (5%) | values, with the prototype as is |

So exact values are possible for ~22% of figures (bar + line charts), exact labels for ~46%
(diagrams), and ~33% stay with the model's eyes.

The same heuristic on the 11 dev figure questions gives 1 bar chart, 1 line chart, 7 diagrams
and 2 pictures. The heuristic is rough and **undercounts charts**: 1706.04269v2 Figure 4
(action-spotting results, 35 filled bars, 18 numbers) was classed as a diagram, probably
because its bars do not start at a zero line.

---

## 4d. Writing and checking the answer (N9, G2, N10b) → Implementation, *Generation and verification*; Results, *RQ1/RQ3* (Day 6, dev, pilot)

### Bar charts read from the PDF (`src/gea/charts.py`, `python scripts/build_indexes.py charts`)

- **How it reads a chart:**
  - each panel's y-axis is calibrated from its tick labels (a straight-line fit; log axes are rejected);
  - each bar's top gives its value, and each bar's rectangle is its box;
  - series names come from the legend swatches, x labels from the tick row, panel names from "(c) …".
- **Accepted:** 7 of the 205 figures, with exact values. Three safety rules keep wrong values out, since G2 would treat them as exact:
  1. line-chart markers resting near zero are not bars (square shapes);
  2. a chart is skipped when most bars have no x label (rotated category labels);
  3. histograms with hundreds of bars stay checkable, but the text shown to the writer is capped.
- **Checked by eye:** RippleNet Fig. 4 (24 bars, e.g. (c) Bing-News hop 4 = 55.8, which the 7B model read as 12); Action Search Fig. 4 (30 bars; this axis starts at 75 and matplotlib clips bars drawn from 0); histograms (1706.00633v4 Fig. 5); a 5-series grouped chart (1705.10667v4 Fig. 2).
- **Rejected after inspection:** the 1809.01246v1 line charts (markers), and 1804.01429v3 Fig. 7 (rotated labels).
- Tests: `tests/test_charts.py`.

### Where G2 stops protecting, with and without the support check (`scripts/g2_boundary.py` → `results/day6_g2_boundary.json`)

The four claim styles of notebook Section 15 on the stub paper, from the template writer, with
the real 7B model as the support checker (2026-10-12):

| claim style | layer 1 (equality tests) | layers 1 + 2 (model support check) |
|---|---|---|
| honest | nothing rejected | nothing rejected (no false alarm) |
| fabricated (a number +1) | rejected (L1) | rejected (L1) |
| uncited number | rejected (L1) | rejected (L1) |
| **misattributed** (a real cell from the **wrong row**) | **missed** | **rejected (L2)**: "+4.4 is the gain for the high-resource subset…" |

**Finding:** the support check closes exactly the gap the equality tests leave, with no false
alarm on the honest draft. This is n = 4 hand-made cases on one stub paper; Day 8 measures it on real answers.

### Real answers on 10 dev questions (`scripts/run_answers.py` → `results/day6_answers.json`, `notes/answers_day6.md`)

All model-driven (N1, N2, N5, G1, N9, G2 layer 2), on real papers. Three prompt versions of the
writer were tried (`notes/prompts.md`):

| writer prompt | claims passing both G2 layers | questions fully verified | rewrites | model calls |
|---|---|---|---|---|
| v4 | 39/51 (76%) | 5/10 | 12 | 95 |
| v5 ("cite listed values as cells") | 24/38 (63%) | 4/10 | 13 | 94 |
| **v6** (final: values marked not quotable; name cleaning) | 34/52 (65%) | 4/10 | 13 | 100 |

The differences are small and noisy on 10 questions; tuning was stopped there to avoid fitting
dev. In every version, 9 of 10 questions ended with at least one verified claim.

**What the answers show (v6; qualitative, the formal scoring is Day 8):**
- Right and fully grounded:
  - "Which dataset has the most 4-hop triples?" → "Bing-News … 6,322,548", boxed on Table 1's cell;
  - ESMM's two auxiliary tasks → "CTR and CTCVR";
  - GRID → "Our method" (= AMIE (Ours)).
- **No answer rather than a wrong one:** "Which model performs best in AUC on MovieLens-1M?" retrieved the wrong table (Table 4, AUC by ripple-set size). Every claim failed G2, and the system said nothing could be stated.
- **The SPIQA gold answer disagrees with the figure:** for "How does the number of common k-hop neighbors change as the hop distance increases…", the gold says it *decreases*. The bar values read from the PDF *increase* (MovieLens-1M with common raters: 4.3, 25.8, 42.6, 59.3), and the agent says "increases", citing those four bars. The gold probably describes panel (d), the ratio. Report gold-label disagreements found during the error analysis.
- Too vague: the faces comparison and "the effect of adding DA" (no numbers).

**Failure modes found in the traces, and the fixes** (all kept by tests):
1. Digits inside names counted as unsourced numbers ("1" in "MovieLens-1M"). Fixed: only standalone numbers are checked (`text.CLAIM_NUMBER`).
2. Rewrites replaced verified claims and got worse. Fixed: a rewrite keeps the claims that passed and replaces only the failed ones.
3. Citation misuse (an "observation" of a paragraph; six values in one cell). Fixed: the JSON schema allows cells only for items with cells, observations only of figures, one value per cell.
4. The writer quoted the "Values" lines it was shown (not paper text). Fixed: they are labelled "cite as cells" and are not quotable.
5. Quote marks copied into cell names ("'Bing-News'"). Fixed: names are cleaned before checking.
6. A heat-map box wholly outside its figure clipped to an empty box. Fixed: it falls back to the figure.

**Proof images:** `data/debug/day6/proof_*.png`, e.g. the three "# 4-hop triples" cells boxed one
colour per claim, and the cited bars of RippleNet Fig. 4 boxed exactly.

### Where do the boxes come from? Exact boxes first, the heat map last (2026-10-12)

The plan's figure boxes came from a heat map for each claim's own words (ColQwen2). That needs
ColQwen2 while the model server runs, which does not fit in this PC's memory. So the replay
first asked how many boxes depended on the map at all. Only 3 of 50 boxes did, and one
verified claim had no box. Three exact sources were then added to N10b (`src/gea/proofs.py`):

- **Caption quotes** are searched in the caption. Captions lie outside a figure's box, so they used to fall back to the map.
- **An observation** ("the figure shows two auxiliary tasks: pCTR and pCTCVR") is boxed on the **diagram labels it names** (`label_box`). A label is one printed line inside the figure, and it counts only when *all* its words occur in the observation. "Auxiliary Tasks" counts here; "Main Task" does not.
- **A zoomed view cited alone** is boxed like its figure. It used to get no box.

A bug was also fixed. In justified text, PDF search returns one line in two pieces when there is a
wide gap, and the second piece was dropped. A caption box stopped at "comparing" instead of the end
of the line.

10 dev answers, replayed from the cache (same 34 verified claims, 100/100 replies cached):

| box source | before | after |
|---|---|---|
| table cell / chart bar (exact) | 11 / 16 | 11 / 16 |
| quoted sentence (exact) | 19 | 19 |
| caption quote (exact) | 0 | 2 |
| diagram labels named by an observation (exact) | 0 | 1 |
| **question's heat map (rough)** | **3** | **1** |
| paragraph (no quote found) | 1 | 1 |
| verified claims with no box | 1 | 0 |

The one map box left is an observation of a **photo** (generated faces, 1812.06589v2 Fig. 5). Photos
have no printed words, so only a heat map can point inside them. Pictures without words are about a
third of the corpus figures (§4c: 67/205 raster), so this case will be common on test. The fix is planned as
a **two-pass run** in the Day 7 runner:

1. Answer with the model server.
2. Stop the server, and cache ColQwen2 vectors of each figure claim's text.
3. Replay from the cache, so each claim's own map gives its box.

Day 4 showed the heat map is noisy inside a page (pointing 15% vs random 8%), so these boxes are
the roughest kind. **For the thesis:** box precision depends on what a claim cites. Cells, bars,
sentences, captions and diagram labels are boxed exactly from the PDF. Only photos fall back to the
learned heat map.

Visual check: `data/debug/day6/` proof images; the ESMM box sits exactly on "Auxiliary Tasks /
pCTCVR / pCTR". Tests: `tests/test_writer.py` (labels, caption, zoom alone).

---

## 4e. Dev error analysis, the baselines and the runner → Results (*error analysis*, RQ1, RQ2 pilot); Experimental Setup (Day 7, dev)

All judged by hand against SPIQA's gold answers (one annotator, dev only; Day 8 scores the test
set formally). Per-question table: `notes/error_analysis_dev.md`. Run files in `results/runs/`.

### Error analysis of the full system (prompt v6, 20 dev questions)

7 right, 4 partly, 6 wrong (incl. vague), 3 no answer. The causes of the 13 not right:

| cause | count |
|---|---|
| G2 too strict: a faithful quote rejected for PDF text quirks ("in- ference", "0 . 5%", ligatures, "item v .") | 4 |
| G1 too lenient: "sufficient" with a fact that does not answer ("DA plays an important role" for "what is the *effect* of DA") | 2 (+1 shared) |
| N9 misread or vague | 3 |
| wrong table or figure chosen | 2 |
| table read wrong (two-level header) | 1 |
| a computed ratio cannot be checked (the model's 185.5 was wrong; true 182.6) | 1 |

Also found: rewrites repeated already-verified claims (up to 3 times); fixed as a bug.

### The two fixes, and one calibration step

1. **G2 layer 1** compares a quote with its evidence after undoing PDF extraction quirks on both
   sides (`text.loose`). It is still word for word: a changed word or number fails.
2. **G1 v7:** "the fact must BE the answer" (named per question type), and the judge writes its
   facts *before* its verdict. llama.cpp fills JSON fields in schema order, so a verdict written
   first was only justified afterwards.

| G1 version (full system, dev) | right | partly | **wrong** | no answer |
|---|---|---|---|---|
| v6 (before) | 7 | 4 | **6** | 3 |
| **v7 (frozen)** | 6 | 1 | **0** | 13 |
| v8 (calibration: "a simple reading of the evidence"; one id) | 5 | 2 | **3** | 10 |

v7 was kept because it is the only version with no wrong answer, and the thesis's priority is
"I could not confirm" over a confident error. Tuning then stopped: v8's differences from v7 are
noise on 20 questions. **The cost:** v7 refuses often, including 2 questions v6 got right. One of
them is the GRID table, where it wanted an explicit statement of the best method.

### The three systems on dev (prompt v7; a pilot for RQ1/RQ2, not the result)

| system | right | partly | wrong | no answer | model calls / question | seconds / question |
|---|---|---|---|---|---|---|
| **Full** | 6 | 1 | 0 | 13 | 8.5 | 39 |
| Single-pass (no planner, no G1, no G2) | 12 | 3 | 5 | 0 | 3.0 | 11 |
| Text-only (text tool only; tables and figures as plain text) | 3 | 1 | 0 | 16 | 8.7 | ~20 |

Seconds are for uncached runs. No node fell back to its rules in any of the 60 runs (`fallbacks` = 0).

- **RQ2 (seeing):** with the same loop and gates, the full system answers twice as many correctly
  as text-only (6 vs 3), and neither is ever wrong.
- **RQ1 (agency):** single-pass answers more questions correctly but is wrong 5 times. Its claims
  are never checked. It also fetches text, a table and a figure in its one round. The full system
  fetches one kind per round, chosen by N1's vote, so a wrong kind ("figure" for a table question)
  costs it rounds.
- This is the System A / System B trade-off of the thesis, already visible on dev. The test set
  (150 questions, 3 systems) measures it with confidence intervals on Day 8.

Making text-only fair: with one tool, N5's rule "never repeat a (sub-goal, tool) pair" made rounds
2 and 3 the same search. In text-only, a repeat may now search again with the planner's new query.
Before this, it refused 18/20 (`text_only_dev_norepeat.jsonl`).

### The first test run, and the fix made on dev (2026-10-15/16) → Results; Discussion

**What happened.** The first frozen test run, with strict G1 v7 alone, abstained on **110 of 142**
questions (77%) before a power cut stopped it. Another 22 had claims withheld, and 10 were fully
answered. The right table or figure *was* in the buffer for about 65% of them. The judge wanted the
answer stated outright.

**What was done (dev only; no test answer was read).** The plan's rule for a severe problem found
during the test run: fix it on dev, re-freeze, re-run all systems on test, and say so.
- **Best effort after the budget** (`cfg.AFTER_BUDGET = "answer"`): G1 decides when to *stop
  searching*. After the last round, if it found answering facts (named and backed) and the rule
  floor holds, the agent answers from them. G2 still checks every claim, and the answer says "the
  evidence found may be incomplete". Otherwise it abstains.
- Also tried and dropped: every tool in round 1, as single-pass does. It gave more evidence and
  worse conclusions.

| full system on dev (20) | right | partly | wrong | no answer |
|---|---|---|---|---|
| strict G1 alone (1st test run's design) | 6 | 1 | 0 | 13 |
| **strict G1 + best effort (re-frozen)** | **9** | 4 | 5 | **2** |
| – confident (G1 convinced) | 6 | 1 | 0 | 1 |
| – best effort (flagged) | 3 | 3 | 5 | 1 |
| + every tool in round 1 (dropped) | 7 | 5 | 6 | 2 |
| single-pass (reference) | 12 | 3 | 5 | 0 |

**For the thesis:**
- Report both test runs: the strict run as an ablation (`full_strict_test.jsonl`) and the re-frozen
  design as the main result.
- The confident/flagged split is a selective-answering result: the system tells the reader which
  answers to trust. On dev, every wrong answer carried the flag.
- Single-pass still answers more questions correctly on dev (12 vs 9). Say this plainly. The
  agent's value is the grounding and the warning, not raw accuracy with a 7B model.

### Problems found while running (Implementation; Discussion, *limitations*)

- **llama-server's RAM prompt cache** (`--cache-ram`, 8 GB by default) grew the server from 7.5 to
  12.7 GB within 30 questions and left 0.8 GB free. Capped at 1 GB, it holds at 8.3 GB.
- **A runaway reply:** G1 once wrote 5,800 tokens of repetition (88 s) inside one JSON string and
  broke off as invalid JSON. Fixed with `maxLength` on G1's strings and `max_tokens` 2048 on every
  call. Question time fell from 120–165 s back to about 35 s.
- **Gemini** (the stronger-model comparison):
  - `gemini-2.5-flash` is closed to new API keys, so the comparison uses `gemini-3.8-flash`;
  - Gemini's endpoint rejects `seed`;
  - it often answers 503 "high demand", so the runner waits and redoes the question.
  - Price (2026-10-12): free tier, or $0.75 / $3.75 per 1M input / output tokens, about $4 for 150
    questions.
  - The free tier's daily quota for 3.8 Flash ran out after about 30 calls (HTTP 429). The full
    run needs about 1,350, and there is no budget for a paid key.
  - **So the comparison uses `gemini-3.5-flash-lite`** (re-frozen 2026-10-13, before any test
    question was run with it). Thesis wording: "a current hosted model (Gemini 3.5 Flash-Lite,
    free tier)", not "a frontier model".
  - If the free daily quota stops it, the runner stops after 5 refused redos (no looping), and
    the next `resume_eval.sh` continues. If it cannot finish all 150, report it on the questions
    it covered, paired with the local full system on the same questions.
  - Smoke test: 3 test questions end to end, 24 calls, no quota error. The one 3.8 Flash answer
    is kept apart (`full_gemini-3.8-flash_test_smoke.jsonl`) and is not part of the comparison.

### The runner (Experimental Setup)

`scripts/run_eval.py` writes one JSON line per question. Each line holds the claims with their
cites, the boxes with page sizes, G1/G2 verdicts and reasons, the path, model calls, fallbacks and
seconds. The line is forced to disk before the next question starts, and a restart skips finished
qids.
- A question during which the model was unreachable (no server, timeout, 429, 5xx) is thrown away
  and redone after waiting, so no stand-in answer enters the results.
- Refused or malformed replies are behaviour of the system and are recorded as `fallbacks`.
- The ColQwen2 vectors of all 150 test questions were cached before llama-server started (36 s).
- On the test split, the runner prints no question or answer text, and refuses to start unless the
  settings equal `eval/frozen_config.json`.

### The five test runs: what each system did (2026-10-16; no answer read) → Results

All 150 test questions per run, from the run files' status fields only. Right/wrong is Day 8.

| system (`results/runs/`) | answered | – confident | – flagged | all claims withheld | abstained | claims kept / drafted | reference in buffer | calls / q | rounds / q | median s / q |
|---|---|---|---|---|---|---|---|---|---|---|
| **full** (`full_test`) | **97** | 22 | 75 | 45 | 8 | 189 / 378 | 109 | 11.6 | 2.72 | 27 |
| full, strict G1 (ablation, `full_strict_test`) | 22 | 22 | 0 | 11 | 117 | 43 / 82 | 109 | 8.4 | 2.72 | 34 |
| single-pass (`single_pass_test`) | 150 | – | – | 0 | 0 | 438 unchecked | 116 | 3.0 | 1 | 11 |
| text-only (`text_only_test`) | 100 | 14 | 86 | 49 | 1 | 210 / 450 | 0* | 12.5 | 2.79 | 42 |
| full, Gemini 3.5 Flash-Lite (`full_gemini_test`) | **121** | 117 | 4 | 5 | 24 | 168 / 191 | 98 | 7.9 | 1.87 | 55** |

"Answered" = at least one claim published. "Confident" = G1 convinced; "flagged" = best effort
after the budget. "All claims withheld" = G2 rejected every claim (the reader sees no answer).
Fallbacks (a node used its rules): full 2 questions, strict 1, text-only 1, others 0. Errors: 0.
\* text-only never retrieves a table or figure as a region, so the reference cannot be in its
buffer by design. \** Gemini's seconds include network and free-tier waits; not comparable.

First reading (to be confirmed by Day 8's scoring):

- **Best effort works as intended:** the strict ablation answered 22 questions; the re-frozen
  design 97. The 22 confident answers are the same number in both runs (the change only touches
  questions where G1 stayed unconvinced).
- **G2 is the bottleneck with the 7B model:** it withheld half of the full system's claims
  (189 of 378 kept), and every claim on 45 questions.
- **A stronger model passes both gates far more often:** with Gemini, G1 was convinced on 117
  questions (vs 22 with the 7B), in fewer rounds (1.87 vs 2.72), and G2 kept 88% of claims
  (vs 50%). Same graph, same prompts, same settings: the gates are not the problem, the 7B's
  reading and quoting are.

**The Gemini run** took three sittings on 2026-10-16 (questions 1–68, 69–125, 126–150). It
stopped twice on "the model could not be reached". The runner counts HTTP 429 (quota) and 503
("high demand") alike, so which of the two stopped it is not known. Each stop lost nothing: the
question was redone. The last 25 questions used a second free-tier API key; the model, the
prompts and the settings were unchanged (the freeze check passed).

**The second pass (2026-10-17): claims about photos boxed by their own map.** 78 such claims
over the five runs. Their ColQwen2 vectors were cached in one pass (about 2 minutes, after the
model server had stopped), then each run was replayed from the model cache with `--rebox`. No
model call was made, so no answer changed, only boxes.

| run | questions reboxed | boxes now from the claim's own map |
|---|---|---|
| full | 19 | 24 |
| single-pass | 31 | 49 |
| text-only | 0 | 0 (it never cites a photo region) |
| full, strict (ablation) | 7 | 9 |
| full, Gemini | 6 | 6 |

The evaluation is complete on 2026-10-17 (commit `6113d04`, "evaluation complete on test").

---

## 4f. Measuring (Day 8, 2026-10-17) → Experimental Setup, *Metrics*; Results

### Definitions used in the scoring (`src/gea/metrics.py`, tests in `tests/test_metrics.py`)

- **Outcome**, one of three per question:
  - **answered** = at least one claim reached the reader. It is then **correct** or **wrong** by the judge.
  - **abstained** = "I could not confirm", or G2 withheld every claim ("nothing can be stated").
- **RF**: the share of the distinct tables and figures in the buffer that are the reference. A zoom
  counts as the figure it enlarges. RF is undefined (not 0) when nothing visual was fetched, so it
  is not defined for text-only.
- **Hit rate**: the reference was in the buffer at all.
- **AES**: hops needed (1, or 2 for the multi-hop subset) divided by search-tool calls (N6a–N6c),
  capped at 1. The automatic zoom (N7) is not a choice and does not count.
- **CGS** per question: the best IoU between any proof box and any hand-drawn answer box on the
  same page ("does a proof point at the answer?"). Three variants are reported:
  - over all annotated questions, with an abstention scoring 0;
  - over answered questions only;
  - the share with IoU ≥ 0.5.
- **Intervals**: 95% bootstrap over questions, 10,000 resamples, seed 42. For differences:
  paired by question, full system minus the other run. "Beyond luck" means the interval
  excludes 0.

### The accuracy judge (`scripts/judge.py`, j1; prompt in `notes/prompts.md`)

- **Setup:** the local Qwen2.5-VL-7B compares each published answer with SPIQA's gold answer. It
  writes its reason first, then the verdict. It sees the answer without the best-effort mark.
- **Dev check** against the earlier hand labels of `notes/error_analysis_dev.md` (v6, v7 and v8;
  34 answered): **agreement 28/34 (82%), Cohen's kappa 0.64.**
  - Every clearly right and every clearly wrong answer matched.
  - All 6 disagreements were answers hand-labelled "partly", which the judge called wrong: a list
    question answered with one of three items, raw counts given instead of the asked ratio,
    "(2,1)" for the gold "2".
  - The judge is stricter than the hand labels about incomplete answers, and defensibly so. It
    was **not tuned** to match them; the formal check is the 30 test answers graded by hand
    (8.3).
- **Dev counts by the judge:**
  - full with best effort: 9 correct / 9 wrong / 2 abstained. The hand labels were 9 right /
    4 partly / 5 wrong / 2, the same once "partly" counts as wrong.
  - single-pass: 11 / 9 / 0 (hand: 12 / 3 / 5).
- **Threat to validity:** the judge is the same model as the evaluated local systems.

### The hand-work sets (`scripts/prepare_annotation.py` → `eval/annotation_set.json`)

- **50 questions for the boxes** (CGS): `random.Random(42)`, 25 with a table and 25 with a figure
  as reference.
  - Drawn from the test split alone, before any system output was looked at.
  - Pages rendered at 144 dpi; boxed in `data/annotate/annotate.html`, an offline page that
    shows the reference region as a dashed outline.
- **30 answers for the judge check**: `random.Random(42)`, 6 published answers from each of the
  five runs, shuffled. The grader sees neither the system nor the judge's verdict, and the same
  rule the judge gets.

### The multi-hop subset (`scripts/tag_multihop.py` → `eval/multihop.json`)

- **The rule:** a question is multi-hop when it asks for a reason or an implication of a
  result ("why", "suggest(s)", "possible explanation"). The result is in the table or figure,
  the reason in the text. The rule reads only the question text.
- **The subset: 10 of 150 questions, all with a table as reference.** It is small, so its
  intervals are wide. Report it as indicative.
- The rule's first version also matched "which explanation method…", which is not a reason
  question, so the rule now asks for "possible explanation".

### Test results, judged by j1 (2026-10-17; before the human check) → Results, RQ1/RQ2; Discussion

`scripts/judge.py` over all 750 test answers (`results/judged/`), then `scripts/score.py` →
`results/summary.json`. **Status:** the judge has not yet been checked against the 30 hand
grades, and CGS and MAP wait for the hand annotations. One power cut during the judge run:
it resumed at question 89 of the first run, with nothing lost.

| system (150 test questions) | correct | wrong | abstained | correct when answering | hit rate | RF | AES | search calls / q |
|---|---|---|---|---|---|---|---|---|
| **full** | 32 (21%) | 65 (43%) | 53 (35%) | 33% | 0.73 | 0.44 | 0.44 | 2.72 |
| single-pass | 51 (34%) | 99 (66%) | 0 | 34% | 0.77 | 0.41 | 0.36 | 3.00 |
| text-only | 23 (15%) | 77 (51%) | 50 (33%) | 23% | 0 | – | 0.42 | 2.79 |
| full, strict G1 (ablation) | 8 (5%) | 14 (9%) | 128 (85%) | 36% | 0.73 | 0.44 | 0.44 | 2.72 |
| full, Gemini 3.5 Flash-Lite | 45 (30%) | 76 (51%) | 29 (19%) | 37% | 0.65 | 0.55 | 0.70 | 1.87 |

**Paired differences** (95% bootstrap; ✔ = beyond luck):

| | correct | wrong | abstained |
|---|---|---|---|
| full − single-pass | −12.7 pts [−22, −4] ✔ | **−22.7 pts [−33, −13] ✔** | +35.3 [+28, +43] ✔ |
| full − text-only | +6.0 [−2, +14] | −8.0 [−18, +2] | +2.0 [−7, +12] |
| full − strict | +16.0 [+11, +22] ✔ | +34.0 [+27, +41] ✔ | −50.0 [−58, −42] ✔ |
| Gemini − full (7B) | +8.7 [0, +17] | +7.3 [−2, +17] | −16.0 [−25, −7] ✔ |

Also beyond luck: AES full 0.44 vs single-pass 0.36. Hit rate full 0.73 vs single-pass 0.77
(−4.7 pts, single-pass fetches every kind every time). Hit rate full vs text-only +73 pts (by
design). Gemini: RF +0.11 and AES +0.26 over the 7B, with fewer rounds.

**Multi-hop subset (10 questions; indicative only):**

| system | correct | wrong | abstained |
|---|---|---|---|
| full | 6 | 2 | 2 |
| single-pass | 4 | 6 | 0 |
| text-only | 2 | 2 | 6 |
| strict | 0 | 0 | 10 |
| Gemini | 4 | 3 | 3 |

Full − single-pass: wrong −40 pts [−70, −10] ✔. Full − text-only: correct +40 pts [+10, +70] ✔.
This is the subset where the plan expected the loop to matter most, but there are only 10
questions.

**Confident vs flagged (the dev finding did not hold on test):**
- full: confident 8 correct / 14 wrong; flagged 24 / 51.
- strict: its 22 answers are the same confident ones, 8 / 14.
- On dev, every wrong answer carried the flag. On test, G1 being convinced does not make a 7B
  answer more likely to be right (36% vs 32%).

**Is abstention selective? (the key Discussion result):** compare single-pass's accuracy on the
questions each system held back with its accuracy on those it answered.

| system | single-pass correct on the questions held back | … on the questions answered |
|---|---|---|
| full (7B) | 40% (of 53) | 31% (of 97) |
| strict (7B) | 36% (of 128) | 23% (of 22) |
| **full with Gemini** | **14% (of 29)** | **39% (of 121)** |

- With the 7B, the gates do not hold back the hard questions: they hold back easier ones about
  as often. The full system answers correctly on 32 questions, single-pass on 51. 17 are correct
  in both, 15 only in full, 34 only in single-pass.
- **With Gemini behind the same graph and prompts, abstention is selective:** it holds back
  mainly the questions that are hard for everyone.
- So the design (search until sufficient, check every claim, abstain otherwise) works when the
  model can judge sufficiency. A 4-bit 7B cannot judge it reliably. This agrees with the gate
  pass rates of §4e (G1 convinced on 117 questions with Gemini vs 22 with the 7B).

**What the thesis can claim (pending the human check of the judge):**
- RQ1: the loop with gates makes the 7B wrong far less often than single-pass (43% vs 66%),
  at the cost of correct answers (21% vs 34%). Its precision when answering is not higher
  (33% vs 34%). The reduction comes from answering less, not from answering better. On
  multi-hop questions it is fewer wrong and more right, but n = 10.
- RQ2: seeing tables and figures (full vs text-only) changes little overall with the 7B: +6 pts
  correct, −8 pts wrong, neither beyond luck. On multi-hop it is +40 pts correct.
- The stronger model makes the gates useful (selective abstention, higher RF and AES). Model
  capacity is the bottleneck, not the architecture: the main Discussion point.
- All wrong rates are high partly because j1 counts incomplete answers as wrong (§ the judge's
  dev check).

### The judge checked against a human grader (2026-10-22; `eval/judge_check.csv`)

30 test answers, 6 per run, graded blind: the grader did not see the system or the verdict, and
had the same rule as the judge.

- **Graded 25 as correct/wrong, 5 "can't tell".**
- **Agreement 22/25 (88%), Cohen's kappa 0.72 (substantial).**
- All 3 disagreements were human correct, judge wrong. The judge never called an answer correct
  that the human called wrong.
- Of the 5 "can't tell": the judge said wrong on 4 and correct on 1. One grader note: "it listed
  the key components correctly but missed the first question" (a two-part question, half
  answered).
- **For the thesis:** the judge is reliable and, if anything, strict. The reported correct rates
  are conservative (the judge undercounts correct answers), and no system's accuracy is
  inflated by it. This agrees with the dev check (28/34, kappa 0.64, all misses on "partly"
  answers).
- `results/summary.json` → `judge_check`.

### The hand-drawn answer boxes: quality control (2026-10-22; `eval/annotations.csv`)

- **Annotator:** Saad Al Abeed, co-author, Computer Science and Engineering, Islamic University of
  Technology. He drew the boxes and graded the 30 answers.
  - He was blind to the system and the judge's verdict when grading, but as a co-author he knew
    the project's aims. Threat to validity: not a fully independent annotator; one annotator.
  - Round 1 of the quality control below was redrawn by the first author, on the cells, bar or
    line part that the gold answer names.
- **First return:** 49 questions boxed and 1 "can't box".
  - 21 boxes covered at least 80% of the whole table or figure, against the guideline.
  - 12 of those are legitimate: the question asks about the whole figure (steps of a pipeline,
    a trend across a chart, reconstructions vs originals).
- **Quality-control round 1:** 9 "which is best / how much better" questions with a whole-table
  box, chosen by a rule (box ≥ 80% of the region and a question naming specific values),
  without looking at any system box.
  - Redrawn on the cells, bar or line part that the gold answer names.
  - Also fixed: the "can't box" question (1811.08257v1#6), whose answer is visible in the figure
    (the activation-layer block).
- **Quality-control round 2:** 7 more questions about one block, label or row, narrowed.
- **Final file:** 50/50 boxed, 91 boxes. 5 legitimate whole-figure boxes remain (#16, 34, 35, 44,
  45). The returns are kept in `eval csv/`.

### Box precision on the 50 annotated questions (RQ3) → Results, RQ3; Discussion

**CGS as defined beforehand** (the best IoU of a proof box with a hand box; abstention = 0):

| system | CGS, all 50 | CGS, answered | IoU ≥ 0.5 |
|---|---|---|---|
| full | 0.07 [0.02, 0.13] | 0.12 (n=29) | 6% |
| single-pass | 0.15 [0.10, 0.22] | 0.15 (n=50) | 6% |
| text-only | 0.00 | 0.01 (n=30) | 0% |
| strict | 0.03 | 0.21 (n=6) | 2% |
| Gemini | 0.05 | 0.06 (n=39) | 2% |

**Why IoU is so low (inspected 2026-10-22):** IoU punishes a correct proof drawn at another size.
- 1704.07121v2#6: the system's boxes are exactly the two right cells, but a cell box spans the
  column (43 pt wide) while the hand box hugs the number (23 pt). IoU 0.43 and 0.46.
- A correct cell inside a hand box around a whole row scores 0.18.
- A correct cell inside a hand box around a table region scores 0.01.
- The coordinates themselves agree; there is no bug.

**Added after seeing the IoU results: the pointing game** (Zhang et al., 2016). A proof hits when
its box's centre lies inside a hand box on the same page. `metrics.pointing`, tested.

| system | pointing, all 50 | pointing, answered | share of a question's boxes that hit |
|---|---|---|---|
| full | 0.16 [0.06, 0.26] | 0.28 [0.14, 0.45] (n=29) | 0.23 |
| single-pass | 0.40 [0.26, 0.54] | 0.40 (n=50) | 0.27 |
| text-only | 0.04 | 0.07 (n=30) | 0.03 |
| strict | 0.06 | 0.50 (n=6) | 0.33 |
| Gemini | 0.20 | 0.26 (n=39) | 0.18 |

Full − single-pass on all 50: −0.24 [−0.36, −0.12] ✔ (the full system answers fewer of them).

**Where the proofs point (post-hoc analysis):** proof boxes on the reference table or figure vs on
other sources.

| system | proof boxes | on the reference | of those, inside the hand box | table cells inside | figure observations inside |
|---|---|---|---|---|---|
| full | 72 | 31 | **22 (71%)** | **12/12** | 8/13 |
| single-pass | 181 | 85 | 53 (62%) | 30/44 | 22/36 |
| Gemini | 68 | 28 | 13 (46%) | 6/6 | 2/4 (+ 5/18 quotes) |

**For the thesis (RQ3):**
- When a claim cites the right table or figure, the box is precise. **Every table-cell proof of
  the full system lands on the answer.** Single-pass misses a third of its cells, because no G2
  checks them.
- The low per-question scores come from upstream. Proofs cite other sources (text quotes, other
  tables: 41 of 72 for full), and many questions are abstained.
- **MAP: the hand check was skipped for time** (2026-10-22). An automatic stand-in is reported
  instead: the share of published claims that cite the question's reference table or figure
  (`metrics.reference_attribution`, over the questions with published claims).
  - full 0.44 [0.34, 0.53] (n=97); single-pass 0.42 [0.36, 0.49] (n=150); text-only 0.00 (it never
    cites a region); strict 0.61 [0.41, 0.80] (n=22); Gemini 0.55 [0.45, 0.63] (n=121).
  - It undercounts claims that rightly cite supporting text. State it as a limitation.
- IoU against a hand box is the wrong yardstick when the two boxes are drawn at different
  granularity. Report CGS as defined, then the pointing game, and say that the second was added
  after seeing the first.
- **Tried: tighter cell boxes (a simulation only; the system was not changed).** Each cited cell's
  box was shrunk to the printed words inside it.
  - CGS: full 0.070 → 0.068, single-pass 0.153 → 0.154, Gemini 0.049 → 0.047.
  - Some questions gain, where the hand box hugs the number. Others lose, where the hand box
    covers a row.
  - So box size does not limit CGS. Abstentions (21 of the 50 annotated questions have no answer
    for full) and proofs on other sources (10 questions point only at other pages) limit it.

---

## 5. The agent design on the stub → Implementation, *design validation* (Day 1)

`paper/tables/tab_execution_profile.tex`, generated by the notebook (test 11I).
R = retrieval rounds, Z = zooms, W = rewrites.

| scenario | R | Z | W | steps | node runs | model calls |
|---|---|---|---|---|---|---|
| Text, 1 round | 1 | 0 | 0 | 11 | 13 | 6 |
| Figure, zoom | 1 | 1 | 0 | 12 | 14 | 6 |
| Running example | 2 | 0 | 0 | 16 | 18 | 8 |
| Running ex., 1 misquote | 2 | 0 | 1 | 19 | 21 | 10 |
| Running ex., never corrected | 2 | 0 | 2 | 23 | 25 | 12 |
| Unanswerable | 3 | 0 | 0 | 20 | 20 | 8 |
| Worst case (3 sub-goals) | 3 | 1 | 2 | 29 | 31 | 14 |

Figures already made: `paper/figures/fig_execution_trace.pdf` (byte-identical on every run)
and `paper/figures/fig_grounded_answer.png`.

---

## 6. Practical constraints → Implementation / Discussion, *limitations*

- **One consumer GPU (12 GB).** Only one big model fits at a time: the 7B language model takes about 7.5 GB, and ColQwen2 about 4.5 GB plus working space. So page images are embedded once, offline, on the GPU. At question time ColQwen2 runs on the CPU, which needs about 5 GB of free RAM.
- **Unreliable power (load shedding, no UPS).**
  - Every expensive file is written to a temporary name, forced to disk, then renamed (`src/gea/safeio.py`).
  - Every long job skips finished work. A power cut costs at most one paper (indexing) or one question (evaluation).
  - Model replies are cached in git (`results/llm_cache`).
- **A shared machine.** Other experiments ran on the same PC. Loading SPECTER2 while memory was nearly full coincided with another job crashing (2026-10-09). Since then every model load first checks free memory and the GPU memory other programs hold (`gea.device.room_problems`).
- **Offline by design.** No model is fetched from the internet at run time.

---

## 7. Still to measure

| result | metric in Methodology | when |
|---|---|---|
| Figure box: heat-map box inside the region vs the whole region, against annotated regions | CGS (RQ3) | Days 6–8, annotated subset |
| Box precision by source: exact (cell, bar, sentence, caption, label) vs heat map (photos, two-pass claim maps) | CGS (RQ3) | Days 7–8 |
| Answer accuracy: agentic loop vs single pass | accuracy, RF, AES (RQ1) | Days 7–8, test |
| Text-only agent baseline | (RQ2) | Days 7–8, test |
| Modality attribution | MAP (RQ2) | Days 7–8, test |
| Box precision on the 50 annotated questions | CGS (RQ3) | Days 6–8 |
| Time and model calls per question on the real backend | AES, efficiency | Days 7–8 |
