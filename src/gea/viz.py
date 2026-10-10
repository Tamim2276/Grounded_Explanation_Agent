# Everything the notebook draws: code listings, the graph, the execution trace and
# the proof on the page.
import inspect
import io
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pymupdf
from matplotlib.patches import FancyBboxPatch, Patch
from PIL import Image as PILImage, ImageDraw

from gea import config as cfg
from gea.corpus import grid_boxes
from gea.profiling import MODEL_NODES
from gea.state import AgentState


# --- code listings -------------------------------------------------------------

def show(*objects) -> None:
    # Show the source of functions, classes or modules from the gea package. The
    # code lives in src/gea/; this puts it on the notebook page next to its
    # explanation, with the file and line to open when you want to change it.
    from IPython.display import Markdown, display
    blocks = []
    for obj in objects:
        lines, start = inspect.getsourcelines(obj)
        path = Path(inspect.getsourcefile(obj)).resolve()
        try:
            path = path.relative_to(cfg.PROJECT_ROOT)
        except ValueError:
            pass
        blocks.append(f"`{path.as_posix()}`, line {max(start, 1)}\n"
                      f"```python\n{''.join(lines).rstrip()}\n```")
    display(Markdown("\n\n".join(blocks)))


# --- the graph -----------------------------------------------------------------

def show_graph(compiled, png: bool = True, mermaid: bool = False):
    # Render a compiled LangGraph.
    #   png=True     try mermaid.ink for an inline image, fall back to ASCII offline
    #   mermaid=True also print the Mermaid source (handy for pasting into the thesis)
    from IPython.display import Image as PNG, display
    g = compiled.get_graph()
    if png:
        try:
            display(PNG(g.draw_mermaid_png(max_retries=3, retry_delay=2.0)))
        except Exception as e:
            print(f"(no PNG: {type(e).__name__} -- falling back to ASCII)\n")
            try:
                print(g.draw_ascii())
            except ImportError:                 # grandalf missing: the source still shows the wiring
                print(g.draw_mermaid())
    else:
        print(g.draw_ascii())
    if mermaid:
        print("\n--- Mermaid source " + "-" * 40)
        print(g.draw_mermaid())


# --- the execution-trace figure (notebook Section 11) ----------------------------

NODE_LABELS = {
    "n1_query_intake":       "N1 Intake",
    "n2_query_encoder":      "N2 Encoder",
    "n5_agentic_planner":    "N5 Planner",
    "n6a_text_retriever":    "N6a Text",
    "n6b_table_retriever":   "N6b Table",
    "n6c_figure_retriever":  "N6c Figure",
    "n7_region_zoom":        "N7 Zoom",
    "n8_evidence_buffer":    "N8 Buffer",
    "g1_sufficiency_check":  "G1 Sufficiency",
    "replan_prep":           "Replan",
    "abstain":               "Abstain",
    "n9_grounded_generator": "N9 Generator",
    "g2_faithfulness_check": "G2 Faithfulness",
    "rewrite_prep":          "Rewrite",
    "drop_unsupported":      "Drop",
    "n10a_claim_links":      "N10a Links",
    "n10b_bounding_boxes":   "N10b Boxes",
    "n10c_dag_trace":        "N10c Trace",
    "n11_report_synthesis":  "N11 Report",
    "n12_final_output":      "N12 Output",
}
CHECK_NODES = {"g1_sufficiency_check", "replan_prep", "abstain",
               "g2_faithfulness_check", "rewrite_prep", "drop_unsupported"}

ROLE_STYLE = {                          # group -> (fill colour, border colour)
    "pipeline": ("#f0efec", "#c3c2b7"),     # neutral grey
    "model":    ("#cde2fb", "#2a78d6"),     # blue
    "check":    ("#fbded2", "#eb6834"),     # orange
}
ROLE_NAMES = {"pipeline": "Pipeline node", "model": "Model call", "check": "Gate or correction loop"}

INK_TEXT, INK_NOTE, INK_MUTED = "#0b0b0b", "#52514e", "#898781"


def node_role(node: str) -> str:
    # Which colour group a node belongs to. Gates are model calls too; they are
    # drawn as checks, because that is their job in the graph.
    if node in CHECK_NODES:
        return "check"
    if node in MODEL_NODES:
        return "model"
    return "pipeline"


def draw_timeline(timeline: list, notes: dict = None, hatched: set = None, save_to=None):
    # Draw a recorded run: one row per step, one box per node.
    #   timeline  the (step, node) list from record_run
    #   notes     {step: text}, a short note written beside that step's box
    #   hatched   {(step, node)}, boxes to hatch, e.g. a rejected draft
    #   save_to   file path; a .pdf path gives a vector figure
    notes, hatched = notes or {}, hatched or set()
    steps = sorted({s for s, _ in timeline})
    per_step = {s: [n for t, n in timeline if t == s] for s in steps}
    n_cols = max(len(v) for v in per_step.values())

    WIDTH, LEFT, TOP, ROW, BOX_H, GAP, LEGEND = 3.35, 0.34, 0.24, 0.21, 0.16, 0.06, 0.42
    col_w = (WIDTH - LEFT - 0.04 - GAP * (n_cols - 1)) / n_cols
    height = TOP + ROW * len(steps) + LEGEND

    with plt.rc_context({"pdf.fonttype": 42, "font.size": 6.5, "hatch.linewidth": 0.6}):
        fig = plt.figure(figsize=(WIDTH, height))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(height, 0)               # y grows downwards, like reading a page
        ax.axis("off")
        ax.text(LEFT - 0.12, TOP - 0.08, "step", ha="right", va="center", color=INK_MUTED)

        for row, s in enumerate(steps):
            y = TOP + row * ROW
            ax.text(LEFT - 0.12, y + BOX_H / 2, str(s), ha="right", va="center", color=INK_MUTED)
            for col, node in enumerate(per_step[s]):
                x = LEFT + col * (col_w + GAP)
                fill, edge = ROLE_STYLE[node_role(node)]
                is_hatched = (s, node) in hatched
                ax.add_patch(FancyBboxPatch((x, y), col_w, BOX_H,
                                            boxstyle="round,pad=0,rounding_size=0.03",
                                            facecolor=fill, edgecolor=edge, linewidth=0.8,
                                            hatch="//////" if is_hatched else None))
                label_bg = dict(facecolor=fill, edgecolor="none", pad=0.8) if is_hatched else None
                ax.text(x + 0.06, y + BOX_H / 2, NODE_LABELS[node], va="center",
                        color=INK_TEXT, bbox=label_bg)
            if s in notes:
                assert len(per_step[s]) < n_cols, f"step {s} has no free space for a note"
                x = LEFT + len(per_step[s]) * (col_w + GAP)
                ax.text(x + 0.02, y + BOX_H / 2, notes[s], va="center", color=INK_NOTE, style="italic")

        handles = [Patch(facecolor=ROLE_STYLE[r][0], edgecolor=ROLE_STYLE[r][1], linewidth=0.8,
                         label=ROLE_NAMES[r]) for r in ROLE_STYLE]
        if hatched:
            fill, edge = ROLE_STYLE["model"]
            handles.append(Patch(facecolor=fill, edgecolor=edge, linewidth=0.8, hatch="//////",
                                 label="Rejected draft"))
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2,
                  frameon=False, fontsize=6, handlelength=1.6, handleheight=1.0, columnspacing=1.2)
        if save_to is not None:
            # No creation date in the PDF: the same run then gives the same bytes, so
            # re-running the notebook does not show up as a changed file in git.
            fig.savefig(save_to, metadata={"CreationDate": None})
    return fig


# --- the proof on the page (notebook Section 13) -------------------------------

def loss_curve_png(epochs: list) -> bytes:
    # Figure 2 of the fake paper: a validation loss that flattens after epoch 10.
    loss = [0.35 + 0.9 * math.exp(-0.35 * (e - 1)) for e in epochs]
    fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=150)
    ax.plot(epochs, loss, marker="o", markersize=3, color="#2C5AA0")
    ax.set(xlabel="Epoch", ylabel="Validation loss", xticks=[1, 5, 10, 15, 20])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def render_paper(paper: dict):
    # Draw a stub paper as a real PDF, every record in its own box.
    doc = pymupdf.open()
    for _ in range(paper["pages"]):
        doc.new_page(width=cfg.PAGE_W, height=cfg.PAGE_H)

    def put(page: int, bbox: tuple, text: str, size: float, font: str, align=pymupdf.TEXT_ALIGN_LEFT):
        rest = doc[page - 1].insert_textbox(pymupdf.Rect(bbox), text, fontsize=size,
                                            fontname=font, align=align)
        assert rest >= 0, f"text does not fit its box: {text[:40]!r}"

    for r in paper["front"]:
        put(r["page"], r["bbox"], r["text"], r["size"], r["font"])
    for h in paper["headings"]:
        put(h["page"], h["bbox"], h["text"], 12, "hebo")
    for c in paper["chunks"]:
        put(c["page"], c["bbox"], c["text"], 10, "helv")
    for t in paper["tables"]:
        boxes = grid_boxes(t)
        shape = doc[t["page"] - 1].new_shape()
        for row in boxes:
            for b in row:
                shape.draw_rect(pymupdf.Rect(b))
        shape.finish(color=(0, 0, 0), width=0.6)
        shape.commit()
        for r, values in enumerate([t["header"]] + t["rows"]):
            for c, value in enumerate(values):
                x0, y0, x1, y1 = boxes[r][c]
                put(t["page"], (x0 + 4, y0 + 4, x1 - 4, y1), value, 9.5, "hebo" if r == 0 else "helv",
                    pymupdf.TEXT_ALIGN_LEFT if c == 0 else pymupdf.TEXT_ALIGN_CENTER)
        put(t["page"], t["caption_bbox"], t["caption"], 9.5, "helv")
    for f in paper["figures"]:
        doc[f["page"] - 1].insert_image(pymupdf.Rect(f["bbox"]), stream=loss_curve_png(f["epochs"]))
        put(f["page"], f["caption_bbox"], f["caption"], 9.5, "helv")
    return doc


CLAIM_COLOURS = [(214, 39, 40), (31, 119, 180), (44, 160, 44), (148, 103, 189), (255, 127, 14)]


def page_image(pdf, page: int, dpi: int = 110) -> PILImage.Image:
    pix = pdf[page - 1].get_pixmap(dpi=dpi)
    return PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)


def draw_proof(state: AgentState, pdf, dpi: int = 110) -> list:
    # One image per page the answer uses: quoted sentences highlighted in yellow
    # (N10a), boxes in one colour per claim (N10b), cropped to the proof.
    s = dpi / 72
    pages = sorted({b["page"] for b in state["boxes"]})
    images = []
    for page in pages:
        image = page_image(pdf, page, dpi).convert("RGBA")
        overlay = PILImage.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        drawn = []
        for claim in state["claims"]:
            for cite in claim["cites"]:
                ev = next(e for e in state["evidence"] if e.id == cite["evidence"])
                if "quote" in cite and ev.kind == "text" and ev.page == page:
                    for r in pdf[page - 1].search_for(cite["quote"]):
                        box = [r.x0 * s, r.y0 * s, r.x1 * s, r.y1 * s]
                        draw.rectangle(box, fill=(255, 214, 0, 95))
                        drawn.append(box)
        per_box = defaultdict(set)
        for b in state["boxes"]:
            if b["page"] == page and "paragraph" not in b["what"]:
                per_box[b["bbox"]].add(b["claim"])
        for (x0, y0, x1, y1), claims in per_box.items():
            for j, claim in enumerate(sorted(claims)):      # a cell used by two claims gets two boxes
                pad = 1 - 4 * j
                box = [x0 * s - pad, y0 * s - pad, x1 * s + pad, y1 * s + pad]
                colour = CLAIM_COLOURS[(claim - 1) % len(CLAIM_COLOURS)]
                draw.rectangle(box, outline=colour + (255,), width=3 if j == 0 else 2)
                drawn.append(box)
        merged = PILImage.alpha_composite(image, overlay).convert("RGB")
        top = max(0, min(b[1] for b in drawn) - 100)      # room for the heading above
        bottom = min(merged.height, max(b[3] for b in drawn) + 40)
        images.append(merged.crop((0, top, merged.width, bottom)))
    return images


def claim_legend(state: AgentState) -> str:
    claims = sorted({b["claim"] for b in state["boxes"] if "paragraph" not in b["what"]})
    return ", ".join(f"claim {n} = rgb{CLAIM_COLOURS[(n - 1) % len(CLAIM_COLOURS)]}" for n in claims)


# --- Day 4: heat maps ------------------------------------------------------------

def heat_overlay(image, heat, box=None, size=None, alpha=0.45):
    # A ColQwen2 heat map (rows x cols) laid over its page image, bright = matches the
    # query best; `box` (PDF points on a page of `size`) is drawn in green.
    import numpy as np
    h = (heat - heat.min()) / (np.ptp(heat) or 1.0)
    rgb = (plt.colormaps["inferno"](h)[..., :3] * 255).astype("uint8")
    layer = PILImage.fromarray(rgb).resize(image.size, PILImage.NEAREST)
    out = PILImage.blend(image.convert("RGB"), layer, alpha)
    if box and size:
        zx, zy = image.size[0] / size[0], image.size[1] / size[1]
        ImageDraw.Draw(out).rectangle([box[0] * zx, box[1] * zy, box[2] * zx, box[3] * zy],
                                      outline=(0, 200, 0), width=6)
    return out
