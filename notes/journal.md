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
