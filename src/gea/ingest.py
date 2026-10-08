# N3 for real papers: read a PDF into the same kind of records as the fake paper
# (BUILD_PLAN.md, Day 2, step 2.2).
#
#     page images   one 150 dpi PNG per page (ColQwen2 reads them on Day 3)
#     captions      lines that start with "Figure 3:", "Fig. 3.", "TABLE II", ...
#     tables        label + caption + the region of the page the table occupies
#     figures       the same for figures
#     chunks        the paper's normal paragraphs, ~150 words each, with section and box
#
# Every record keeps its page and its box (x0, y0, x1, y1) in PDF points, origin at the
# top-left corner -- the boxes N10b draws on Day 6.
#
# How a table or figure is found. PyMuPDF's own table finder misses many tables, and a
# figure can be one picture, hundreds of picture tiles or pure line drawings. Captions,
# however, are always there. So each region starts from its caption: look in the strip
# of page next to it -- above a figure caption, below a table caption, as CS papers do --
# up to the nearest normal paragraph, and take everything graphical in that strip:
# pictures, line drawings and small text (cell values, axis labels).
import re
from collections import Counter

import pymupdf

from gea.safeio import atomic_path, atomic_write_json

DPI = 150
CHUNK_WORDS = 150
PAD = 2                     # points of margin around every table and figure region
MARGIN = 40                 # top and bottom page margins: page numbers and running headers

# "Figure 3:", "Fig. 3.", "FIGURE 3 |", "Table 2:", or IEEE style "TABLE II" alone on its
# line with the title on the next -> kind, number
CAPTION = re.compile(r"^(Figure|Fig\.|FIGURE|Table|TABLE)\s*([0-9]+|[IVXLC]+)\s*([:.|]|$)")
# "3 Results", "4.2 Ablation", "A Proofs" (single line, no full stop at the end)
HEADING = re.compile(r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\.?\s+[A-Z][^.]{1,70}$")
# unnumbered headings, kept as named sections
NAMED_HEADING = re.compile(r"^(abstract|keywords|acknowledge?ments?|appendix|appendices)$", re.I)
REFERENCES = re.compile(r"^(\d+\.?\s+)?(references|bibliography)$", re.I)


# --- geometry -------------------------------------------------------------------------

def overlap_x(a, b) -> bool:
    # Do two boxes share some horizontal range?
    return min(a[2], b[2]) - max(a[0], b[0]) > 2


def overlap_y(a, b) -> bool:
    # Do two boxes sit at the same height (share at least half the smaller one)?
    h = min(a[3], b[3]) - max(a[1], b[1])
    return h > 0.5 * min(a[3] - a[1], b[3] - b[1])


def inside(a, b, share: float = 0.5) -> bool:
    # Does at least `share` of box a lie inside box b?
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = (a[2] - a[0]) * (a[3] - a[1])
    return w > 0 and h > 0 and area > 0 and w * h >= share * area


def union(boxes) -> tuple:
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def rounded(box) -> tuple:
    return tuple(round(v, 1) for v in box)


# --- step 1: the text of a page, as blocks made of rows --------------------------------

def clean(text: str) -> str:
    # One space between words: justified text arrives as pieces with spaces of their own.
    return re.sub(r"\s+", " ", text).strip()


def join_lines(lines: list) -> str:
    # A block's lines as one text, words broken at a line end joined again:
    # "be-" + "tween" -> "between" (only before a lowercase letter, so "GPT-" + "4" and
    # "Fig-" + "Net" stay as they are).
    text = ""
    for line in lines:
        if re.search(r"[A-Za-z]-$", text) and line[:1].islower():
            text = text[:-1] + line
        else:
            text = f"{text} {line}" if text else line
    return clean(text)


def rows_of(lines: list, size: float) -> list:
    # Group a block's lines into rows: lines at the same height are one row. A row is
    # "gapped" when its pieces are more than one font size apart -- the cells of a table
    # row (a space between words is a third of that), whether PyMuPDF
    # made them separate lines or separate pieces (spans) of one line -- and not when it
    # is ordinary running text.
    rows = []
    for ln in sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0])):
        if rows and overlap_y(rows[-1]["parts"][0]["bbox"], ln["bbox"]):
            rows[-1]["parts"].append(ln)
        else:
            rows.append({"parts": [ln]})
    for r in rows:
        parts = sorted(r["parts"], key=lambda l: l["bbox"][0])
        gaps = [b["bbox"][0] - a["bbox"][2] for a, b in zip(parts, parts[1:])]
        r.update(bbox=union(p["bbox"] for p in parts), text=clean(" ".join(p["text"] for p in parts)),
                 gapped=any(g > size for g in gaps) or any(p["gapped"] for p in parts))
    return rows


def blocks_of(page) -> list:
    # Every text block on the page: its box, its text (hyphenated line breaks joined),
    # its most common font size and its rows.
    flags = pymupdf.TEXTFLAGS_DICT | pymupdf.TEXT_DEHYPHENATE
    blocks = []
    for b in page.get_text("dict", flags=flags)["blocks"]:
        if b["type"] != 0:                                  # 0 = text, 1 = image
            continue
        lines, sizes = [], Counter()
        for line in b["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            text = clean(" ".join(s["text"] for s in spans))
            if text:
                # a line whose pieces are far apart is a table row inside one line
                gaps = [q["bbox"][0] - p["bbox"][2] for p, q in zip(spans, spans[1:])]
                big = max(s["size"] for s in spans)          # wider than one font size: not a space
                lines.append({"bbox": tuple(line["bbox"]), "text": text,
                              "gapped": any(g > big for g in gaps)})
            for s in line["spans"]:
                sizes[round(s["size"] * 2) / 2] += len(s["text"])
        if not lines:
            continue
        size = sizes.most_common(1)[0][0]
        blocks.append({"bbox": tuple(b["bbox"]), "text": join_lines([l["text"] for l in lines]),
                       "lines": len(lines), "size": size, "rows": rows_of(lines, size)})
    return blocks


def body_font_size(doc) -> float:
    # The font size most of the paper's characters are written in -- the paragraphs.
    sizes = Counter()
    for page in doc:
        for b in blocks_of(page):
            sizes[b["size"]] += len(b["text"])
    return sizes.most_common(1)[0][0]


# --- step 2: captions and headings ----------------------------------------------------

def caption_label(text: str):
    # ("table" | "figure", "Table 2") for a caption, or None.
    m = CAPTION.match(text)
    if not m:
        return None
    kind = "table" if m.group(1).lower().startswith("tab") else "figure"
    return kind, f"{kind.capitalize()} {m.group(2)}"


def split_caption(block: dict):
    # Find a caption inside a block. PyMuPDF sometimes merges a caption with the table
    # rows under it, or with a figure's labels above it, into one block. So the caption
    # is found row by row: the row that starts with "Table 2:" -- if it is the block's
    # first row, or follows rows that are not running text -- plus the rows that
    # continue it. It ends at the first gapped row (a table row) or a big vertical gap.
    # Returns (caption, kind, label, other_rows) or None.
    rows = block["rows"]
    for i, row in enumerate(rows):
        label = caption_label(row["text"])
        if not label:
            continue
        before = rows[i - 1] if i else None
        if before and not before["gapped"] and before["bbox"][2] - before["bbox"][0] > \
                0.6 * (block["bbox"][2] - block["bbox"][0]):
            continue                                   # mid-paragraph: "... in\nTable 2. We ..."
        j = i + 1
        while j < len(rows) and not rows[j]["gapped"] and \
                rows[j]["bbox"][1] - rows[j - 1]["bbox"][3] <= block["size"]:
            j += 1
        caption = {"bbox": union(r["bbox"] for r in rows[i:j]),
                   "text": " ".join(r["text"] for r in rows[i:j])}
        return caption, label[0], label[1], rows[:i] + rows[j:]
    return None


def heading_of(block: dict, body_size: float):
    # The section a heading block starts ("4", "4.2", "A", "abstract", "references"),
    # or None if the block is not a heading.
    if block["lines"] > 2 or block["size"] < body_size - 0.25:
        return None
    text = block["text"].strip()
    if REFERENCES.match(text):
        return "references"
    if NAMED_HEADING.match(text):
        return text.lower()
    m = HEADING.match(text)
    return m.group(1) if m else None


# --- step 3: where the tables and figures are ------------------------------------------

def graphics_of(page, has_table_caption: bool) -> tuple:
    # Boxes of everything graphical on the page -- pictures and groups of line drawings
    # -- and, if a table caption is on the page, PyMuPDF's own table guesses.
    # Returns (graphics, table_guesses); the guesses are part of the graphics too.
    rect = page.rect
    boxes = [tuple(i["bbox"]) for i in page.get_image_info()]
    boxes += [tuple(r) for r in page.cluster_drawings()]
    tables = [tuple(t.bbox) for t in page.find_tables().tables] if has_table_caption else []
    # drop slivers (rules, underlines) and anything off the page
    keep = lambda b: (b[2] - b[0] > 3 and b[3] - b[1] > 3
                      and b[0] >= rect.x0 - 1 and b[3] <= rect.y1 + 1)
    tables = [b for b in tables if keep(b)]
    return [b for b in boxes if keep(b)] + tables, tables


def adjacent_guess(caption: dict, guesses: list, side: str):
    # PyMuPDF's table guess that touches the caption on `side` ("below": starts at most
    # 30 points under it; "above": ends at most 30 points over it), or None. Guesses
    # that touch no table caption -- often the axes of a plot -- are never used.
    cap = caption["bbox"]
    if side == "below":
        near = [g for g in guesses if overlap_x(g, cap) and 0 <= g[1] - cap[3] <= 30]
        return min(near, key=lambda g: g[1] - cap[3], default=None)
    near = [g for g in guesses if overlap_x(g, cap) and 0 <= cap[1] - g[3] <= 30]
    return min(near, key=lambda g: cap[1] - g[3], default=None)


def table_side(pages: list) -> str:
    # Where this paper puts its tables relative to their captions: "below" (caption on
    # top, the usual habit) or "above" (caption underneath). A paper keeps one habit,
    # so for every table caption see on which side content starts closer -- a table
    # guess, or any item (cell text, rules) -- and let the paper's captions vote.
    votes = Counter()
    for p in pages:
        for cap, kind, _ in p["captions"]:
            if kind != "table":
                continue
            c = cap["bbox"]
            near = [i for i in p["items"] if overlap_x(i, c)]
            gap_below = min((i[1] - c[3] for i in near if i[1] >= c[3] - 1), default=99)
            gap_above = min((c[1] - i[3] for i in near if i[3] <= c[1] + 1), default=99)
            if min(gap_below, gap_above) <= 20:
                votes["below" if gap_below <= gap_above else "above"] += 1
    return "above" if votes["above"] > votes["below"] else "below"


def area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def crosses(a, b) -> bool:
    # Do two boxes overlap with a positive area?
    return min(a[2], b[2]) - max(a[0], b[0]) > 0.5 and min(a[3], b[3]) - max(a[1], b[1]) > 0.5


def grow(box, text_items: list, walls: list, reach_x: float = 15, reach_y: float = 12) -> tuple:
    # Extend a table box over table text right next to it: PyMuPDF's table finder often
    # sees only the ruled columns and misses a "Method" column or a header row beside
    # them -- or splits one table into several guesses. Text and the other guesses are
    # added within a line's reach (never a picture, so a figure beside the table stays
    # out), and never across a wall (a paragraph or a caption).
    changed = True
    while changed:
        changed = False
        for t in text_items:
            if inside(t, box, 0.99):
                continue
            near = (t[0] <= box[2] + reach_x and t[2] >= box[0] - reach_x and
                    t[1] <= box[3] + reach_y and t[3] >= box[1] - reach_y)
            bigger = union([box, t])
            if near and not any(crosses(bigger, w) for w in walls):
                box, changed = bigger, True
    return box


def region_of(caption: dict, kind: str, page_rect, walls: list, items: list,
              side: str = "below") -> tuple:
    # The region of the table or figure that `caption` belongs to.
    #   walls  boxes the region may not cross: normal paragraphs and other captions
    #   items  boxes that can be part of a region: graphics and small text
    #   side   for tables: where this paper puts tables relative to their captions
    # Returns (box, how): how = "content" if found from items, "strip" if the strip was empty.
    cap = caption["bbox"]
    top, bottom = page_rect.y0 + MARGIN, page_rect.y1 - MARGIN
    others = [w for w in walls if w != cap]
    above = max([w[3] for w in others if overlap_x(w, cap) and w[3] <= cap[1] + 1] + [top])
    below = min([w[1] for w in others if overlap_x(w, cap) and w[1] >= cap[3] - 1] + [bottom])
    strips = {"above": (above, cap[1]), "below": (cap[3], below)}
    if kind == "figure":
        order = ["above", "below"]                 # figure captions sit under their figure
    else:
        order = [side, "above" if side == "below" else "below"]
    content = {}
    for s in order:
        y0, y1 = strips[s]
        found = [i for i in items if overlap_x(i, cap) and i[1] >= y0 - 2 and i[3] <= y1 + 2]
        content[s] = union(found) if found else None
    first, second = content[order[0]], content[order[1]]
    # The usual side wins -- unless it holds only a speck while the other side holds a
    # real table or figure (a stray label is not a table).
    if first and not (second and area(first) < 0.1 * area(second)):
        return first, "content"
    if second:
        return second, "content"
    y0, y1 = strips[order[0]]                      # nothing found: the preferred strip
    return (cap[0], y0, cap[2], y1), "strip"


# --- step 4: the whole paper ------------------------------------------------------------

def read_page(page, body_size: float, section: str) -> dict:
    # Pass 1 for one page: sort its text into captions, headings and normal blocks
    # (each block remembering the section it is in), and collect its graphics.
    p = {"page": page.number + 1, "rect": page.rect, "captions": [], "normal": [],
         "loose": [], "headings": []}                 # loose: rows split off caption blocks
    for b in blocks_of(page):
        found = split_caption(b)
        if found:
            caption, kind, label, rest = found
            p["captions"].append((caption, kind, label))
            p["loose"] += [r["bbox"] for r in rest]
            continue
        heading = heading_of(b, body_size)
        if heading:
            section = heading
            p["headings"].append({"page": p["page"], "section": section, "text": b["text"],
                                  "bbox": rounded(b["bbox"])})
            continue
        b["section"] = section
        p["normal"].append(b)
    p["graphics"], p["table_guesses"] = graphics_of(
        page, any(kind == "table" for _, kind, _ in p["captions"]))
    p["section_after"] = section

    # A paragraph is body-size text, several lines or wide, not lying on a graphic, and
    # not table-like. Paragraphs are walls a region may not cross; everything else that
    # is not in the page margins (page numbers, running headers) can be part of a region.
    rect = page.rect
    p["is_paragraph"] = [b["size"] >= body_size - 0.6
                         and (b["lines"] >= 2 or b["bbox"][2] - b["bbox"][0] > 0.35 * rect.width)
                         and not table_like(b)
                         and not any(inside(b["bbox"], g) for g in p["graphics"]) for b in p["normal"]]
    p["walls"] = [b["bbox"] for b, par in zip(p["normal"], p["is_paragraph"]) if par] + \
                 [c["bbox"] for c, _, _ in p["captions"]]
    in_margin = lambda box: box[3] <= rect.y0 + MARGIN or box[1] >= rect.y1 - MARGIN
    p["text_items"] = [i for i in p["loose"] +
                       [b["bbox"] for b, par in zip(p["normal"], p["is_paragraph"]) if not par]
                       if not in_margin(i)]
    p["items"] = [g for g in p["graphics"] if not in_margin(g)] + p["text_items"]
    return p


NUMBER_TOKEN = re.compile(r"^[-+±]?[\d.,]+%?$")


def table_like(block: dict) -> bool:
    # Rows of a table that PyMuPDF did not attach to any caption: most rows have cells
    # side by side, or most words are numbers ("VAGER 0.8556 0.5292 0.9271 ...").
    rows = block["rows"]
    words = block["text"].split()
    gapped = sum(r["gapped"] for r in rows) / len(rows)
    numbers = sum(bool(NUMBER_TOKEN.match(w)) for w in words) / max(1, len(words))
    return gapped >= 0.5 or numbers >= 0.4


def place_regions(p: dict, body_size: float, side: str) -> tuple:
    # Pass 2 for one page: the region of every table and figure, and the page's
    # paragraphs that lie outside all of them. Returns (records, paragraphs).
    rect, normal, is_paragraph = p["rect"], p["normal"], p["is_paragraph"]
    walls, items = list(p["walls"]), p["items"]

    # Tables first: a table right next to its caption, on the side this paper puts its
    # tables, as PyMuPDF found it, is the most exact region there is. Each guess serves
    # one caption, and it becomes a wall, so a figure stacked against it cannot swallow it.
    records, unused = [], list(p["table_guesses"])
    for cap, kind, label in sorted(p["captions"], key=lambda c: c[1] != "table"):
        guess = adjacent_guess(cap, unused, side) if kind == "table" else None
        if guess:
            unused.remove(guess)
            box, how = grow(guess, p["text_items"] + unused, walls), "table-finder"
            walls.append(box)
        else:
            box, how = region_of(cap, kind, rect, walls, items, side)
            if how == "strip" and kind == "table":
                # Last resort, when nothing at all was found (a table whose rules PyMuPDF
                # cannot read, and whose text looks like a paragraph): only captions and
                # headings are walls, so the table's text may count as content.
                soft_walls = [c["bbox"] for c, _, _ in p["captions"]] + [h["bbox"] for h in p["headings"]]
                text = [b["bbox"] for b, par in zip(normal, is_paragraph) if par]
                box, how = region_of(cap, kind, rect, soft_walls, items + text, side)
                how = "last-resort" if how == "content" else how
        box = (max(rect.x0, box[0] - PAD), max(rect.y0, box[1] - PAD),      # a little air, so a
               min(rect.x1, box[2] + PAD), min(rect.y1, box[3] + PAD))      # crop never clips text
        # Doubtful: found only by the last resort or not at all, or smaller than 1% of
        # the page. Day 4 trusts ColQwen2's heat map more than such a region.
        doubtful = how in ("strip", "last-resort") or area(box) < 0.01 * rect.width * rect.height
        records.append({"kind": kind, "label": label, "page": p["page"], "bbox": rounded(box),
                        "found_by": how, "doubtful": doubtful, "caption": cap["text"],
                        "caption_bbox": rounded(cap["bbox"])})
    paragraphs = [(p["page"], b) for b, par in zip(normal, is_paragraph)
                  if par and not any(inside(b["bbox"], r["bbox"]) for r in records)]
    return records, paragraphs


def ingest_pdf(pdf_path, out_dir) -> dict:
    # Read one paper. Writes the page images, then corpus.json LAST, as the "this paper
    # is finished" mark: after a power cut, a paper without corpus.json is redone.
    doc = pymupdf.open(pdf_path)
    body_size = body_font_size(doc)
    paper = {"paper_id": pdf_path.stem, "pdf": str(pdf_path), "body_font_size": body_size,
             "pages": [], "headings": [], "tables": [], "figures": [], "chunks": []}

    # Pass 1: page images, and every page's captions, headings, text and graphics.
    pages, section = [], "front"                      # before the first heading: title, abstract
    for page in doc:
        n = page.number + 1
        image = out_dir / f"page_{n}.png"
        with atomic_path(image) as tmp:
            page.get_pixmap(dpi=DPI).save(tmp)
        paper["pages"].append({"page": n, "size": (page.rect.width, page.rect.height),
                               "image": image.name})
        p = read_page(page, body_size, section)
        section = p["section_after"]
        paper["headings"] += p["headings"]
        pages.append(p)

    # Pass 2: one habit for the whole paper, then every page's regions and paragraphs.
    side = table_side(pages)
    paper["table_side"] = side
    paragraphs = []
    for p in pages:
        records, page_paragraphs = place_regions(p, body_size, side)
        for r in records:
            kind = r.pop("kind")
            paper[kind + "s"].append({"id": f"{kind[0]}{len(paper[kind + 's']) + 1}", **r})
        paragraphs += page_paragraphs

    paper["chunks"] = make_chunks(paragraphs, body_size)
    atomic_write_json(out_dir / "corpus.json", paper)
    return paper


def make_chunks(paragraphs: list, body_size: float) -> list:
    # Join paragraphs in reading order into ~CHUNK_WORDS-word chunks. A new chunk starts
    # on a new page or section, in a different column (left edges not aligned: the
    # full-width title above a half-width abstract), after a big vertical gap (a table
    # or figure in between) or when the chunk is full. So a chunk's box stays one tidy
    # rectangle of text; `boxes` keeps every paragraph's own box.
    chunks, current, last = [], None, None
    for page, b in paragraphs:
        if b["section"] == "references":
            continue                                  # the bibliography answers nothing
        box = b["bbox"]
        same = (current is not None and current["page"] == page
                and current["section"] == b["section"]
                and abs(box[0] - last[0]) < 12 and overlap_x(box, last)
                and box[1] - last[3] < 3 * body_size
                and len(current["text"].split()) < CHUNK_WORDS)
        if not same:
            current = {"id": f"c{len(chunks) + 1}", "page": page, "section": b["section"],
                       "bbox": box, "boxes": [], "text": ""}
            chunks.append(current)
        current["text"] = join_lines([t for t in (current["text"], b["text"]) if t])
        current["boxes"].append(rounded(box))
        current["bbox"] = rounded(union([current["bbox"], box]))
        last = box
    return chunks


# --- checking it by eye -------------------------------------------------------------------

COLOURS = {"chunks": (31, 119, 180), "tables": (44, 160, 44), "figures": (214, 39, 40),
           "captions": (255, 127, 14), "headings": (148, 103, 189)}


def draw_page(paper: dict, page_dir, n: int):
    # The page image with every record drawn on it: blue chunks, green tables, red
    # figures, orange captions, purple headings. Returns a PIL image.
    from PIL import Image, ImageDraw
    image = Image.open(page_dir / f"page_{n}.png").convert("RGB")
    s = image.width / paper["pages"][n - 1]["size"][0]           # points -> pixels
    draw = ImageDraw.Draw(image)

    def box(b, colour, width=3, label=None):
        draw.rectangle([b[0] * s, b[1] * s, b[2] * s, b[3] * s], outline=colour, width=width)
        if label:
            draw.text((b[0] * s + 4, b[1] * s + 2), label, fill=colour)

    for c in paper["chunks"]:
        if c["page"] == n:
            box(c["bbox"], COLOURS["chunks"], 2, f"{c['id']} sec {c['section']}")
    for h in paper["headings"]:
        if h["page"] == n:
            box(h["bbox"], COLOURS["headings"], 2)
    for kind in ("tables", "figures"):
        for r in paper[kind]:
            if r["page"] == n:
                box(r["bbox"], COLOURS[kind], 4, f"{r['label']} ({r['found_by']})")
                box(r["caption_bbox"], COLOURS["captions"], 2)
    return image
