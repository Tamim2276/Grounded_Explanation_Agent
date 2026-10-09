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
