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
  - The free tier's daily quota ran out after about 30 calls (HTTP 429). The full run needs about
    1,350, so it needs a paid key. The runner stops after 5 refused redos instead of looping.
  - The smoke test answered 1 test question end to end with Gemini.

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
