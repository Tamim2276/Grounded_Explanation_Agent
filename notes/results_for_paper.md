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

Source: `results/day4_regions.json`, from `python scripts/eval_regions.py` (9 s, 2026-10-10).
The reference's kind decides the tool (table → N6b, figure → N6c). A hit means the reference
is ranked 1st (hit@1) or in the top 3 among all tables or figures of its paper.

| | "text" ranking: hit@1 | hit@3 | random hit@1 |
|---|---|---|---|
| tables (9) | **89%** (8/9) | **100%** | 35% |
| figures (11) | **73%** (8/11) | 82% | 19% |
| all (20) | **80%** | 90% | 26% |

"text" = N6a's hybrid ranking applied to the caption (SPECTER2) and to the caption **plus the
words printed inside the region** (keyword): a table's cells, a chart's axis labels and legend.

**Ablation, captions only:** tables 6/9 first, figures 8/11. **Adding the region's words lifted
tables to 8/9.** Example: "Which dataset has the most 4-hop triples?" matches Table 1's
*cells*; its caption only says "Basic statistics of the three datasets."

**Still to measure, once ColQwen2 can load:** the "visual" (heat inside the region) and "both"
rankings, and whether the heat map points at the reference (pointing accuracy, and IoU of the
heat-map box with the region). On 2026-10-10 another job held the GPU and memory, so ColQwen2
could not run. The same script measures all of this when it can.

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
  - All 5 ran end to end in 0.1–7.6 s each, and the reference table or figure reached the evidence buffer in 4 of 5.
  - The answers are template sentences until the language model arrives (Days 5–6).

### Heat-map orientation

The saved patch vectors are read as a rows × columns grid, row by row. This is confirmed from
the code: Qwen2-VL's image processor orders tokens as (row block, column block). A model-free
check, predicting each patch's ink from its vector, was too weak to decide (correlation 0.12 vs
0.08): ColQwen2's patch vectors encode meaning, not ink. The empirical check is the pointing/IoU
measurement above.

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
| Heat maps: "visual"/"both" region choice, pointing accuracy, heat-map box IoU (`scripts/eval_regions.py`) | (supports RQ2, CGS) | next time ColQwen2 fits (B580 free) |
| Answer accuracy: agentic loop vs single pass | accuracy, RF, AES (RQ1) | Days 7–8, test |
| Text-only agent baseline | (RQ2) | Days 7–8, test |
| Modality attribution | MAP (RQ2) | Days 7–8, test |
| Box precision on the 50 annotated questions | CGS (RQ3) | Days 6–8 |
| Time and model calls per question on the real backend | AES, efficiency | Days 7–8 |
