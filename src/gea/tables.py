# Table structure (BUILD_PLAN.md Day 4, step 4.3): every table region from N3 becomes a
# grid of cells, each with its text and its box on the page.
#
# The Table Transformer (TATR) is an object detector: on an image of the table it draws a
# box around every ROW and every COLUMN. A cell is where a row and a column cross. The
# cell's text is then read from the PDF's own text layer at that spot -- exact, with no
# OCR -- so every value keeps the box N10b will draw.
import json
from functools import lru_cache
from pathlib import Path

import pymupdf
import torch

from gea.indexes import local_snapshot
from gea.safeio import atomic_write_json

TATR = "microsoft/table-structure-recognition-v1.1-all"
DPI = 150            # the table image TATR looks at
THRESHOLD = 0.6      # keep detections at least this confident
MARGIN = 6.0         # points of white space around the region: TATR expects some
TABLES_DIR = "tables"  # data/corpus/<paper>/tables/<table id>.json, one per table


@lru_cache(maxsize=1)
def tatr():
    # The Table Transformer, from the local download only; small (29M weights), CPU.
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    path = local_snapshot(TATR)
    proc = AutoImageProcessor.from_pretrained(path)
    # The download's config gives only {"longest_edge": 1000}, a newer format than our
    # transformers reads; the model was trained with the longest side at 1000 px.
    proc.size = {"shortest_edge": 1000, "longest_edge": 1000}
    return proc, TableTransformerForObjectDetection.from_pretrained(path).eval()


def detect(page: pymupdf.Page, region: tuple) -> list:
    # TATR's (label, box in PDF points, confidence) for one table region.
    from PIL import Image
    clip = pymupdf.Rect(region) + (-MARGIN, -MARGIN, MARGIN, MARGIN)
    clip &= page.rect
    pix = page.get_pixmap(dpi=DPI, clip=clip)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    proc, model = tatr()
    with torch.inference_mode():
        out = model(**proc(images=image, return_tensors="pt"))
    det = proc.post_process_object_detection(out, threshold=THRESHOLD, target_sizes=[image.size[::-1]])[0]
    scale = 72 / DPI
    return [(model.config.id2label[int(label)],
             (clip.x0 + b[0] * scale, clip.y0 + b[1] * scale, clip.x0 + b[2] * scale, clip.y0 + b[3] * scale),
             float(s))
            for label, b, s in zip(det["labels"], det["boxes"].tolist(), det["scores"])]


def distinct(boxes: list, axis: int) -> list:
    # Sort row (axis 1) or column (axis 0) boxes and drop near-duplicates: TATR sometimes
    # returns two boxes for one row, and a doubled row would double every value.
    boxes = sorted(boxes, key=lambda b: b[axis])
    kept = []
    for b in boxes:
        if kept:
            prev = kept[-1]
            overlap = min(prev[axis + 2], b[axis + 2]) - max(prev[axis], b[axis])
            if overlap > 0.5 * min(prev[axis + 2] - prev[axis], b[axis + 2] - b[axis]):
                continue
        kept.append(b)
    return kept


def cell_text(words: list, box: tuple) -> str:
    # The words whose CENTRE lies in the box, in reading order. (Taking every word that
    # merely touches it would also take the next row's: TATR's row boxes overlap a little.)
    inside = [w for w in words if box[0] <= (w[0] + w[2]) / 2 <= box[2] and box[1] <= (w[1] + w[3]) / 2 <= box[3]]
    lines = []                                   # words whose centres are within 3 pt share a line
    for w in sorted(inside, key=lambda w: (w[1] + w[3]) / 2):
        if lines and abs((w[1] + w[3]) / 2 - lines[-1][0]) <= 3:
            lines[-1][1].append(w)
        else:
            lines.append([(w[1] + w[3]) / 2, [w]])
    return " ".join(w[4] for _, line in lines for w in sorted(line, key=lambda w: w[0]))


def read_cells(page: pymupdf.Page, rows: list, cols: list) -> tuple:
    # cell = the column's x-range x the row's y-range; its text from the PDF text layer
    grid = [[(c[0], r[1], c[2], r[3]) for c in cols] for r in rows]
    words = page.get_text("words")
    return grid, [[cell_text(words, b) for b in row] for row in grid]


def tatr_cells(page: pymupdf.Page, region: tuple) -> tuple:
    found = detect(page, region)
    rows = distinct([b for name, b, _ in found if name == "table row"], axis=1)
    cols = distinct([b for name, b, _ in found if name == "table column"], axis=0)
    return read_cells(page, rows, cols)


def rule_cells(page: pymupdf.Page, region: tuple) -> tuple:
    # The fallback: PyMuPDF's own table finder inside the region (ruled tables only).
    tabs = page.find_tables(clip=pymupdf.Rect(region))
    if not tabs.tables:
        return [], []
    t = max(tabs.tables, key=lambda t: t.row_count * t.col_count)
    words = page.get_text("words")
    grid = [[tuple(c) if c else None for c in row.cells] for row in t.rows]
    return grid, [[cell_text(words, b) if b else "" for b in row] for row in grid]


def usable(grid: list, text: list) -> bool:
    # At least 2 rows and 2 columns, and some text in the cells.
    return (len(grid) >= 2 and min(len(r) for r in grid) >= 2
            and sum(bool(v) for row in text for v in row) >= 2)


def parse_table(doc: pymupdf.Document, record: dict) -> dict:
    # One table record -> {"header", "rows", "cell_boxes", "method"}: the stub's table
    # format, so table_text() and the cells list work unchanged. Header = first row;
    # a row's name = its first cell.
    page = doc[record["page"] - 1]
    for method, find in (("tatr", tatr_cells), ("find_tables", rule_cells)):
        grid, text = find(page, tuple(record["bbox"]))
        if usable(grid, text):
            boxes = [[tuple(round(v, 1) for v in b) if b else None for b in row] for row in grid]
            return {"id": record["id"], "method": method, "header": text[0], "rows": text[1:],
                    "cell_boxes": boxes}
    # No grid (not a real table, or TATR found one column): keep the region's raw text,
    # so the generator still reads the values -- only the per-cell boxes are missing.
    return {"id": record["id"], "method": "none", "header": [], "rows": [], "cell_boxes": [],
            "text": " ".join(page.get_textbox(pymupdf.Rect(record["bbox"])).split())}


def table_file(folder: Path, table_id: str) -> Path:
    return Path(folder) / TABLES_DIR / f"{table_id}.json"


def build_table_cache(folder: Path, device: str = "cpu") -> int:
    # Parse every table of one paper that is not parsed yet; one file per table, written
    # whole, so a power cut costs at most the table in progress. Returns tables parsed.
    folder = Path(folder)
    paper = json.loads((folder / "corpus.json").read_text(encoding="utf-8"))
    corpus_time = (folder / "corpus.json").stat().st_mtime        # N3 re-read the paper: parse again
    todo = [t for t in paper["tables"] if not table_file(folder, t["id"]).exists()
            or table_file(folder, t["id"]).stat().st_mtime < corpus_time]
    if todo:
        with pymupdf.open(paper["pdf"]) as doc:
            for t in todo:
                atomic_write_json(table_file(folder, t["id"]), parse_table(doc, t))
    return len(todo)


def tables_are_current(folder: Path) -> bool:
    folder = Path(folder)
    paper = json.loads((folder / "corpus.json").read_text(encoding="utf-8"))
    corpus_time = (folder / "corpus.json").stat().st_mtime
    return all(table_file(folder, t["id"]).exists()
               and table_file(folder, t["id"]).stat().st_mtime >= corpus_time for t in paper["tables"])


def load_tables(folder: Path) -> dict:
    # table id -> parsed table, for the tables parsed so far
    folder = Path(folder) / TABLES_DIR
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")}


def draw_cells(doc: pymupdf.Document, record: dict, parsed: dict, highlight=None):
    # The table region at 150 dpi with every cell box drawn; `highlight` = (row, col)
    # index of one cell to draw in green (the cell a claim cites).
    from PIL import Image, ImageDraw
    page = doc[record["page"] - 1]
    clip = (pymupdf.Rect(record["bbox"]) + (-MARGIN, -MARGIN, MARGIN, MARGIN)) & page.rect
    pix = page.get_pixmap(dpi=DPI, clip=clip)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw, z = ImageDraw.Draw(image), DPI / 72
    for r, row in enumerate(parsed["cell_boxes"]):
        for c, b in enumerate(row):
            if b:
                xy = [(b[0] - clip.x0) * z, (b[1] - clip.y0) * z, (b[2] - clip.x0) * z, (b[3] - clip.y0) * z]
                hit = highlight == (r, c)
                draw.rectangle(xy, outline=(0, 170, 0) if hit else (220, 40, 40), width=4 if hit else 1)
    return image
