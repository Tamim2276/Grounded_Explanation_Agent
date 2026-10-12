- 2026-10-08: llama.cpp build 11476 (commit 988190680), Vulkan build; B580 visible, 12190 MiB
- 2026-10-08, Day 2 (N3, reading real papers):
  - Questions: scripts/select_questions.py, seed 42, whole papers per split, at most 4 (dev) / 8 (test)
    questions per paper -> dev 20 questions / 5 papers (9 table, 11 figure); test 150 / 31 papers
    (71 table, 79 figure). 7 of SPIQA's 666 references use Roman numerals ("TableII").
  - Reader: src/gea/ingest.py; scripts/build_corpus.py reads 36 papers (475 pages) in ~90 s on the CPU.
  - Coverage of the answer-holding table/figure: dev 20/20, test 150/150 (results/day2_coverage.json).
    Regions: 63 from PyMuPDF's table finder (grown), 300 from the caption strip, 1 last resort; 3 of 364 doubtful.
  - What broke and how it was fixed (each kept by tests/test_ingest.py):
    - caption glued to table rows in one PyMuPDF block -> captions found per row; stop at a gapped row;
      cell gaps can be only ~11 pt, so "gapped" = more than one font size
    - two regions grabbed the same content (table stacked on a figure) -> tables placed first, as walls
    - find_tables misses/splits tables, catches only the ruled columns -> used only next to a table caption,
      then grown over neighbouring table text and other guesses (12 pt reach)
    - some papers put table captions under the tables -> per-paper vote, two passes over the paper
    - table rows taken as paragraphs -> "table-like" = mostly gapped rows or >= 40% numbers
    - page number taken as a table -> top/bottom 40 pt margins ignored; a speck loses to real content
    - IEEE "TABLE I" with the title on the next line -> caption may end right after the number
    - sections assigned once per page -> per paragraph; "front" before the first heading
    - "be- tween" -> words broken at a line end joined (before a lowercase letter)
  - Known limitations: "state-of-the-" + "art" -> "state-of-theart"; 1706.00633v4 draws Table 2 and
    Figure 2 in one frame (Figure 2 region doubtful); one table's rules invisible to PyMuPDF (last resort).
- 2026-10-09, Day 3 (N4 indexes, N6a text search):
  - 3.2 SPECTER2 encoder: src/gea/indexes.py, embed_text(texts, kind="doc"|"query"). Loads from the
    local HF download only (local_files_only), ~4 s; CPU, ~4 real chunks/s (with OrdinalFed training
    running on the same machine). tests/test_indexes.py: 6 tests.
  - Finding: SPECTER2 matches topics, not answers. 4 hand-made questions x (answer, word-sharer,
    unrelated): unrelated always last; answer above word-sharer 2 of 4 (e.g. "few labelled examples":
    answer 0.769, word-sharer 0.779, unrelated 0.733). Scores bunch between 0.65 and 0.78: only the
    order means anything. Keyword search cannot tell the answer from the unrelated text (both 0).
  - Memory: the PC is shared (OrdinalFed training queue, other projects); RAM 15.6 GB + 15 GB page file,
    free commit fell to 0.6-2 GB. A SPECTER2 test run hit MemoryError at the moment an OrdinalFed run
    crashed (02:35; its queue resumed it, one round lost). Since then every model load checks first:
    gea.device.room_problems() (free memory + GPU memory other programs hold); the SPECTER2 tests skip
    below 2.5 GB free.
  - 3.3 text index: scripts/build_indexes.py text --gpu -> 36 papers, 2,532 texts (chunks + captions),
    0.9 min on the B580 (~10 min on the CPU). text.faiss (IndexFlatIP) + text_ids.json per paper.
  - 3.4 page index: scripts/build_indexes.py pages -> 475 pages in 3.0 min on the B580 (~0.3 s/page;
    planned 1-2 h). ColQwen2 = colqwen2-base + v1.0 LoRA merged, bf16, loaded from local files only.
    A 150-dpi page (1275x1650) -> 672x868 -> 31 x 24 = 744 patch vectors of 128 (bf16, row by row);
    ~1.6 MB per paper. Patch ~25.5 pt (9 mm) square on a 612x792 pt page.
  - 3.6 first retrieval numbers (results/day3_retrieval.json, dev, 20 questions, 20 s on the B580):
    page hit@1 75% (random 12%), page hit@3 95% (random 37%), text hit@3 75%.
    Only miss outside the top 3: 1708.00160v2#1 (Figure 2, page 7, rank 10 of 15).
    Dev Q 1803.03467v4#0 (4-hop triples, Table 1 p.6): page 6 first, but close (13.48 vs 13.27 for p.7).
  - Text hit@3 checked again: of the 15 lenient hits, 5 = the reference's caption, 5 = a paragraph naming
    the label, 5 = only on the same page (can be coincidence). Strict text hit@3 = 10/20 = 50%; report that.
    eval_retrieval.py now records text_match and text_hit@3_strict. Undercount example: 1804.07931v2#1
    (clicks vs impressions) -- the top-3 chunk c13 states the answer but is on page 2, not Figure 1's page.
  - 3.5 real N6a: retrieval.real_text_retriever (cfg.BACKEND="real"); n6a node unchanged. Returns the top
    cfg.TEXT_TOP_K=3 paragraphs (caption rows excluded), section filter = the section and its subsections,
    falling back to the whole paper. get_corpus("real") now loads text.faiss (refuses a stale one).
    Ranking compared on dev (scripts/compare_text_search.py -> results/day3_text_search.json):
    strict@3 keyword 70%, SPECTER2 35%, hybrid (RRF, k=60) 60%; strict@1 45 / 15 / 50%.
    Keyword wins because SPIQA questions reuse the paper's exact terms. Chosen: hybrid (best @1, 2/20 behind
    keyword @3, robust to reworded planner queries). Revisit on Day 5 with planner sub-queries.
  - Slip: a sed range delete truncated scripts/eval_retrieval.py (end pattern did not match -> deleted to EOF);
    restored from git and re-applied with exact edits. Rule: no sed range deletes.
  - 3.7 notebook Section 18 (6 cells, tag day3-cell): index contents, page search for dev 1803.03467v4#0 from the
    saved scores (page 6 first, Table 1 boxed), live hybrid N6a (c46 "names Table 1" in 3rd place), Day 3 numbers.
    Whole notebook runs (130 cells, 33 s). 17C kept as Day 2's keyword search so it loads no model.
  - Keyword mode now needs no model and no index (no SPECTER2 tie-break); compare_text_search re-run: numbers
    unchanged (45/70/90%). Model loads in tests/notebook skip the memory check when SPECTER2 is already loaded.
- 2026-10-10, Day 4 (N6b, N6c, N7 real):
  - PC busy: another project's training held 7 GB of the B580 and memory was 4-5 GB free all day, so
    ColQwen2 (~5 GB) could not load. Built so it degrades: page_query_vectors() returns cached vectors,
    or computes them when ColQwen2 fits (B580 free, else >= 7 GB RAM), else None -> tables/figures are
    chosen by their words alone and figure boxes fall back to the N3 region.
  - 4.3 tables: src/gea/tables.py; Table Transformer (v1.1-all) on 150-dpi crops, CPU; cell = row x column;
    text from the PDF words whose CENTRE is in the cell (get_textbox took the next row's words: TATR row
    boxes overlap); words grouped into lines within 3 pt (fixed "et al. [2017] Chung"). The download's
    preprocessor config ({"longest_edge": 1000}) needs size set by hand for transformers 4.47.
    scripts/build_indexes.py tables: 159 tables in 0.9 min; 157 by TATR, 2 without a grid (a boxed text
    example; one table TATR saw as 1 column) -> raw region text kept. Checked by eye: 3 dev tables match.
  - Orientation (4.2): a model-free check (patch vectors vs page ink) was too weak (corr 0.12 vs 0.08);
    instead read Qwen2-VL's image processor: tokens are ordered (row block, column block) -> view(rows, cols)
    is right. The empirical check (pointing / IoU) runs with ColQwen2: scripts/eval_regions.py.
  - Bug found: hybrid rank fusion gave credit to items a search did not find (keyword score 0 -> tied
    2nd); with 5 tables SPECTER2's slight preference for Table 5 beat the only keyword match (Table 1).
    Fixed: score 0 earns nothing. Day 3 text numbers re-run: unchanged.
  - Captions alone missed Table 1 ("Basic statistics ..." vs "4-hop triples" in its cells): tables and
    figures are now scored on caption + the words printed inside their region.
  - 4.5 zoom: PNG from the PDF at <= 300 dpi, long side <= 1024 px; route_zoom uses the real page size.
  - Results: scripts/eval_regions.py (text only so far): right table/figure 1st 80% (random 26%), top 3
    90%; tables 8/9, 9/9; figures 8/11, 9/11. Caption-only ablation: tables 6/9 -> 8/9 with region words.
  - 4.6 smoke: scripts/smoke_real.py, 5 dev questions through the whole graph (rules planner, 4 with
    cfg.SUB_GOAL_MODALITY set): all ran, reference reached the buffer in 4/5, 0.1-7.6 s each.
  - 4.7 notebook Section 19 (135 cells, 47 s). Tests: 84.
- 2026-10-11, Day 4 finished with the B580 free (user ran scripts/eval_regions.py; query vectors cached):
  - First result: the heat map did not point at the reference (pointing 15%, random 8%; map-box IoU 0.07),
    and choosing regions by the heat inside them was worse than words alone (75% vs 80%).
  - Investigated: orientation is right (reference hotter than the rest by +0.035 row-major vs +0.016 swapped;
    per-word best patches land on Table 1's rows). Max over content words / mean / z-mean: no better (10%).
    The map lights up the question's words wherever they are -- figure, caption, the paragraph about it,
    margins (data/debug/day4/figure4_heat.png, heat_tokens_p6.png).
  - Changed: "visual" = ColQwen2's MaxSim score of the region's PAGE (what it is trained for); the heat only
    narrows the box inside the chosen region, clipped to it. Dev, right table/figure 1st: text 80%, page 80%,
    both 85% (default), heat 55%, text+heat 75%; top 3: 90 / 95 / 95 / 90 / 90%. Tables by page score: 9/9.
    N10b box vs whole region: IoU median 0.81. Tests 85; notebook Section 19 shows the heat map.
- 2026-10-11, Day 5 (model in N1, N2, N5, G1):
  - llama-server reserves ~7.5 GB of Windows commit memory (RAM + page file; ~4.3 GB in RAM) whatever the flags
    (--no-host, --load-mode mmap: no change). With another project's job (tools.mask_quality, ~5.6-7 GB) running,
    free commit fell to 0.6-0.8 GB, so the server was stopped each time (twice it started in a brief window).
    start_llm.sh now refuses below 10 GB free commit and names the biggest users. On Windows, stopping the bash
    wrapper does not stop llama-server.exe (taskkill //IM llama-server.exe //F). Lasting fix: a page file on D:.
  - llm.call_llm: temperature 0, seed, JSON schema, cache results/llm_cache/<2 hex>/<sha256>.json with the prompt
    text, model, seconds, usage; PROMPT_VERSION in the key; cfg.LLM_CACHE switch for tests. N1/N2/G1 behind
    cfg.LLM_MODE ("rules"|"llm"), N5 behind PLANNER_MODE; each falls back to rules and says so in the trace.
  - N1 v1 6/20 kinds right (figures 1/11) -> v2 with examples 8/20; regex 2/20. N2: names only if in the question;
    "Table 2"-style refs stay regex. G1 v1 lenient (GRID) -> v3 must name item + fact; code checks item and numbers.
  - Traces: 10 dev, 42 calls, 147 s; rerun 6.9 s (cache); power-cut test (server stopped): 10/10 identical, an
    uncached question falls back. Rule floor fixed the GRID question (gold answer named by G1).
  - Tools now search with sub-goal words + the planner's query (the model's queries are short keyword lists).
  - ColQwen2 page score now uses the QUESTION's cached vectors (computed before the server starts): it cannot load
    beside llama-server on this PC.
  - Chart reading probe: 2/2 wrong (12 vs ~55; 25 vs ~4) on the 4-panel Figure 4 crop.
  - Known glitch: one chunk's section is "L" (1812.06589v2) -- a heading mis-read on Day 2; cosmetic.
  - Tests: 95 (10 new, model-free with a fake server). Notebook Section 20 replays a model run from the cache.
  - N1's kind of evidence by vote (src/gea/kinds.py): the 9 most similar of 469 labelled SPIQA questions from the
    82 test-A papers outside dev/test (SPECTER2 question vectors, cached in data/cache/kind_examples.npz, rebuilt
    if the splits change). Dev: 17/20 (model v2 8/20, regex 2/20). Real backend only (stub keeps its rules).
    10 traces: right kind 4/10 -> 9/10, right table/figure in the buffer 3/10 -> 7/10, 0 abstained. Tests: 98.
  - Decision (user, 2026-10-11): no fine-tuning of the 7B; keep the local 7B as the evaluated system; add a
    Gemini API run as a "stronger model" comparison on Day 7 (switch in the evaluation runner).
  - Figure weak spot (asked by the user): prototype reads vector bar charts from the PDF itself (tick labels give
    each panel's y-axis by a straight-line fit; each filled rectangle's top gives its value): Figure 4 of
    1803.03467v4, all 22 bars of panels (a)-(c) right, e.g. (c) hop 4 = 55.8 (model 12), (a) hop 1 = 4.3 (model 25).
    First try failed: panels (a) and (c) share an axis x-position, so their tick columns merged -> split a column
    where values stop falling. Corpus figures (rough heuristic): 46% vector diagrams, 33% raster, 17% vector line
    charts, 5% vector bar charts; it undercounts charts (1706.04269v2 Fig. 4 missed). Prototype in the scratchpad;
    to become src/gea/charts.py with tests on Day 6.
- 2026-10-12: plan updated -- the user wants a website after the thesis: Days 11-13 (static showcase from the run files on GitHub Pages first, optional live demo); Days 6-7 now save proofs/run records website-ready.
- 2026-10-12, Day 6 (N9, G2, N10b real):
  - charts.py: vector bar charts read from the PDF (tick-label calibration, bar tops, legend names, x labels,
    panel names). Bugs found by looking: panels sharing an axis x merged (split columns where values rise);
    axis starting at 75 with bars drawn from 0 (use the visible floor); axis titles taken as x labels (first
    row only, >= 2 pt below the axis); line-chart markers taken as bars (square shapes, unclipped size); rotated
    labels (skip charts without x labels); garbled multi-line legends (2.5 pt line tolerance). 7 charts kept.
  - Bars become cells, so the writer cites them like table cells, G2 checks them exactly, N10b boxes the bar.
  - N9 model writer (schema per call: id enum, cell/observation only where valid), rewrite keeps verified
    claims; G2 layer 2 support check (cfg.SUPPORT_CHECK auto/on/off); N10a handles observations; N10b boxes
    quotes where printed (sentence, diagram label; first occurrence + wrapped lines) and records page size + cite.
  - g2_boundary.py: support check catches the misattributed claim, no false alarm (stub, real model).
  - run_answers.py on 10 dev: v4 39/51 claims kept, v5 24/38, v6 34/52 (kept v6); fixes from the traces:
    standalone-number rule, keep verified claims, cite kinds by schema, values lines not quotable, name
    cleaning, empty clipped box. SPIQA gold for 1803.03467v4#1 says "decreases"; the PDF's bars increase.
  - Proofs in data/debug/day6/. Notebook Section 21 (146 cells). Tests: 112.
  - Lesson: backslashes in bash heredocs get mangled in this environment -- use the editor for code edits.
- 2026-10-12, Day 6 follow-up (the user asked whether the per-claim heat-map limitation can be overcome):
  - Counted first: only 3/50 boxes of the 10 dev answers used the question's map, and 1 verified claim had none.
  - Exact sources added to N10b: caption quotes searched in the caption (outside the figure box); an observation
    boxed on the diagram labels it names (a printed line counts when all its words occur in the observation);
    a zoomed view cited alone boxed like its figure. Bug found by looking: justified text splits one line into
    two search hits and the second was dropped (a caption box stopped at "comparing").
  - Replay from the cache (100/100 cached, same 34 claims): map boxes 3 -> 1, unboxed claims 1 -> 0. The one left
    is a photo (no printed words): the Day 7 runner gets a two-pass step (answer; stop the server; cache ColQwen2
    vectors of figure claims; replay) for per-claim maps. Tests: 115.
- 2026-10-12, Day 7 (baselines, runner, error analysis):
  - ColQwen2 vectors of the 150 test questions cached before llama-server (36 s on the B580;
    scripts/cache_question_vectors.py; the runner refuses to start without them).
  - 7.2 variants: cfg.VARIANT + graph.build_graph; single_pass (N1 -> N2 -> every tool once with the
    question -> N8 -> N9 -> proofs; no N5, G1, G2); text_only (planner offered "text" only, sub-goals
    forced to text, tables and figures searchable as plain text: caption + printed words). The full
    system's planner prompt stays byte-identical. 6 stub tests.
  - 7.3 runner scripts/run_eval.py: one JSON line per question (gea/runs.record, website-ready),
    resume by qid, a question during which the model was unreachable (no server, timeout, 429, 5xx)
    is thrown away and redone after waiting; refused/bad replies are recorded as "fallbacks"; crashes
    recorded as status "error"; test split prints no text and must match eval/frozen_config.json;
    --rebox second pass (claim maps, scripts/claim_vectors.py). Gemini switch (cfg.LLM_PROVIDER;
    key from env or .env; no seed: HTTP 400; gemini-2.5-flash closed to new keys -> gemini-3.8-flash;
    frequent 503 "high demand", handled by the wait-and-redo loop).
  - 7.1 error analysis (notes/error_analysis_dev.md): v6 7 right / 4 partly / 6 wrong / 3 none. Top
    causes: G2 too strict on PDF quirks (4), G1 too lenient (3). Fixes: text.loose for G2 layer 1;
    G1 v7 (the fact must be the answer; facts before verdict). Bug: rewrites duplicated kept claims.
    v7: 6 / 1 / 0 / 13. v8 (calibration) 5 / 2 / 3 / 10 -> dropped; v7 kept, tuning stopped.
  - Found while running: llama-server's prompt cache in RAM (--cache-ram, default 8 GB) grew the
    server from 7.5 to 12.7 GB and left 0.8 GB free -> --cache-ram 1024 (holds at 8.3 GB). G1 once
    generated 5,800 tokens (88 s) of repetition -> maxLength on G1's strings, max_tokens 2048.
    The proof drawing crashed on 4 claims citing one line (nested rings inverted) -> rings outwards.
