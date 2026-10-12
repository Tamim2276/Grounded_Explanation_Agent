# Dev error analysis (Day 7.1, 2026-10-12)

The full system on the 20 dev questions, judged by hand against SPIQA's gold answer (one
annotator; the formal scoring of the test set is Day 8). Run files: `results/runs/full_dev_v6.jsonl`
(before), `full_dev_v7.jsonl` (= `full_dev.jsonl`, chosen), `full_dev_v8.jsonl` (tried, dropped).

Outcomes: **right**, **partly** (right but incomplete), **wrong** (incl. vague answers that do not
answer), **none** (abstained, or every claim withheld by G2).

| question | reference | v6 | cause in v6 | v7 | v8 |
|---|---|---|---|---|---|
| 1803.03467v4#0 most 4-hop triples | Table 1 | right | – | right | right |
| 1803.03467v4#1 k-hop neighbours vs hops | Figure 4 | right* | (*the gold says "decreases"; the bars in the PDF increase) | none | none |
| 1803.03467v4#2 best model by AUC | Table 3 | none | wrong table chosen (Table 4); G1 lenient | none | wrong |
| 1803.03467v4#3 role of ripple sets | Figure 2 | partly | G2 too strict ("item v ." in the PDF text) | none | none |
| 1812.06589v2#0 realistic faces | Figure 6 | wrong (vague) | wrong figure chosen (Figure 5) | none | wrong |
| 1812.06589v2#1 effect of adding DA | Table 3 | wrong (vague) | G1 lenient: the table was never retrieved | none | none |
| 1812.06589v2#2 best on GRID | Table 2 | right | (a two-level header read as "Evaluation on") | none | none |
| 1812.06589v2#3 frame discriminator | Figure 2 | right | (the same claim 3 times: rewrite bug) | right | right |
| 1804.07931v2#0 ESMM auxiliary tasks | Figure 2 | right | – | right | right |
| 1804.07931v2#1 clicks vs impressions | Figure 1 | partly | G2 too strict ("in- ference") | none | none |
| 1804.07931v2#2 clicks per conversion | Table 1 | partly | a computed ratio cannot be checked (185.5; the true ratio is 182.6) | none | none |
| 1804.07931v2#3 best on CVR and CTCVR | Table 2 | right | – | right | right |
| 1708.00160v2#0 probability of a pattern | Figure 4 | wrong | N9: needs reasoning over the counts | none | none |
| 1708.00160v2#1 support of "ana=NAM" | Figure 2 | right | – | right | partly ("(2,1)") |
| 1708.00160v2#2 best on CoNLL, significant? | Table 4 | none | table read wrong (two-level header: cells missing) | none | none |
| 1708.00160v2#3 best on WikiCoref | Table 2 | wrong | N9: metric names taken for model names | none | none |
| 1706.04269v2#0 three components | Figure 3 | partly | G2 too strict ("ex- tracted") | partly | right |
| 1706.04269v2#1 fewest observations | Figure 4 | none | G2 too strict ("0 . 5%") | right | partly |
| 1706.04269v2#2 temporal context | Figure 5 | wrong (restates) | N9 vague | none | wrong |
| 1706.04269v2#3 effect of training size | Table 1 | wrong | G1 lenient: the table was never retrieved | none | none |

## Totals

| version | right | partly | wrong | none |
|---|---|---|---|---|
| v6 (before) | 7 | 4 | 6 | 3 |
| **v7 (chosen)** | 6 | 1 | **0** | 13 |
| v8 (tried) | 5 | 2 | 3 | 10 |

## Causes in v6 (the 13 not right)

| cause | count | fixed on Day 7? |
|---|---|---|
| G2 too strict: a faithful quote rejected for PDF text quirks | 4 | **yes** (`text.loose`) |
| G1 too lenient: "sufficient" with a fact that does not answer | 2 (+1 shared) | **yes** (G1 v7) |
| N9 misread or vague | 3 | no (three different causes) |
| wrong table or figure chosen | 2 | no |
| table read wrong (two-level header) | 1 | no |
| computed number (a ratio) not checkable | 1 | no (by design) |

Plus a bug, not a category: a rewrite repeated the claims it was told were kept (fixed).

## What was decided

- The plan's rule: fix the two most frequent causes, then stop tuning.
- G1 v7 overshot: no wrong answers, but 13 abstentions. It refused the GRID table because it did not "explicitly state" the best method. One calibration step (v8) was tried: it allowed "a simple reading of the evidence" and required one evidence id. It brought back 3 wrong answers and still refused the same questions.
- **v7 was kept**: it is the only version without a wrong answer. The thesis's priority is to say "I could not confirm" rather than be wrong. Tuning stopped there; v8's differences are noise on 20 questions.
- The cost to report: v7 abstains often, including on 2 questions that v6 got right (the bar chart and GRID). Day 8 measures this on test, where single-pass gives the comparison.
