# The PDF reader (N3, BUILD_PLAN.md Day 2): its rules on small hand-made examples, and
# two real pages it once got wrong. The real-page tests skip when data/ is not built.
import json

import pytest

from gea import config as cfg
from gea.ingest import (caption_label, heading_of, join_lines, rows_of, split_caption,
                        table_like)

CORPUS = cfg.DATA_DIR / "corpus"


def block(lines: list, size: float = 9.0) -> dict:
    # A text block as blocks_of() builds it, from (x0, y0, x1, y1, text, gapped) lines.
    parts = [{"bbox": l[:4], "text": l[4], "gapped": l[5]} for l in lines]
    rows = rows_of(parts, size)
    box = (min(l[0] for l in lines), min(l[1] for l in lines),
           max(l[2] for l in lines), max(l[3] for l in lines))
    return {"bbox": box, "text": join_lines([l[4] for l in lines]), "lines": len(lines),
            "size": size, "rows": rows}


@pytest.mark.parametrize("text, expected", [
    ("Figure 3: Validation loss per epoch.", ("figure", "Figure 3")),
    ("Fig. 3. Validation loss", ("figure", "Figure 3")),
    ("FIGURE 3 | Results", ("figure", "Figure 3")),
    ("Table 2 : Out-of-domain evaluation", ("table", "Table 2")),
    ("TABLE II", ("table", "Table II")),                       # IEEE: title on the next line
    ("Figure 3 shows the training loss", None),               # a sentence, not a caption
])
def test_caption_labels(text, expected):
    assert caption_label(text) == expected


def test_a_caption_stops_at_the_first_table_row():
    # PyMuPDF merged the caption with the table under it (1803.03467v4, page 6).
    b = block([(323, 195, 552, 204, "Table 2: Hyper-parameter settings for", False),
               (323, 205, 450, 214, "the three datasets.", False),
               (323, 216, 378, 225, "MovieLens-1M", False),           # first cell ...
               (389, 216, 552, 225, "d = 16, H = 2", False)])         # ... and the next, 11 pt away
    caption, kind, label, rest = split_caption(b)
    assert (kind, label) == ("table", "Table 2")
    assert caption["text"] == "Table 2: Hyper-parameter settings for the three datasets."
    assert len(rest) == 1 and rest[0]["gapped"]


def test_a_caption_word_inside_a_paragraph_is_not_a_caption():
    b = block([(60, 100, 535, 110, "and the results of the second experiment are reported in", False),
               (60, 111, 535, 121, "Table 2. We then compare the two models on the test set.", False)])
    assert split_caption(b) is None


def test_table_rows_are_not_paragraphs():
    numbers = block([(60, 100, 535, 110, "VAGER 0.8556 0.5292 0.9271 0.6491", False),
                     (60, 111, 535, 121, "LR 0.7705 0.3994 0.8885 0.5882", False)])
    words = block([(60, 100, 535, 110, "We train for 20 epochs with early stopping on the", False),
                   (60, 111, 535, 121, "validation set, as described in Section 3.", False)])
    assert table_like(numbers) and not table_like(words)


@pytest.mark.parametrize("text, expected", [
    ("3 Results", "3"), ("4.2 Ablation study", "4.2"), ("A Proofs", "A"),
    ("References", "references"), ("Abstract", "abstract"), ("ACKNOWLEDGMENTS", "acknowledgments"),
    ("We train the model.", None),
])
def test_headings(text, expected):
    assert heading_of({"text": text, "lines": 1, "size": 10.0}, body_size=10.0) == expected


def test_words_broken_at_a_line_end_are_joined():
    assert join_lines(["the correlation be-", "tween the two"]) == "the correlation between the two"
    assert join_lines(["a GPT-", "4 model"]) == "a GPT- 4 model"        # not before a lowercase word


def paper(pid: str) -> dict:
    path = CORPUS / pid / "corpus.json"
    if not path.exists():
        pytest.skip(f"{pid} not read yet (python scripts/build_corpus.py)")
    return json.loads(path.read_text(encoding="utf-8"))


def test_two_tables_on_one_page_get_their_own_regions():
    tables = {r["label"]: r["bbox"] for r in paper("1803.03467v4")["tables"] if r["page"] == 6}
    assert tables["Table 1"] != tables["Table 2"]
    assert 200 < tables["Table 2"][1] and tables["Table 2"][3] < 245     # just the three rows


def test_a_paper_with_captions_under_its_tables_is_recognised():
    p = paper("1812.06589v2")
    assert p["table_side"] == "above"
    table3 = next(r for r in p["tables"] if r["label"] == "Table 3")
    assert table3["bbox"][3] < table3["caption_bbox"][1]                  # the table is above its caption
