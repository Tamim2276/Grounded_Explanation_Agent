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
