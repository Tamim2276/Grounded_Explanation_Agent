# Day 4 numbers (BUILD_PLAN.md Day 4): do the real N6b / N6c pick the right table or
# figure, and do ColQwen2's heat maps point at it? On the 20 dev questions.
#
#   python scripts/eval_regions.py
#
# Each question's reference kind decides the tool (table -> N6b, figure -> N6c):
#   hit@1, hit@3  the reference is the 1st / among the first 3 of its kind in the paper
#   by method     "text"   caption + the words inside the region
#                 "visual" ColQwen2's heat inside the region
#                 "both"   the two combined by rank (cfg.REGION_SEARCH)
#   pointing      the hottest patch of the reference's page lies inside the reference
#   map IoU       the box around the page's hottest patches vs the reference region (N3)
# "visual", "both", pointing and IoU need ColQwen2's query vectors: computed when ColQwen2
# fits on this PC (B580 free, or ~7 GB of memory), cached in data/cache/page_queries/, and
# otherwise reported as unavailable. Writes results/day4_regions.json.
import sys
import time

import numpy as np

from gea import config as cfg
from gea.corpus import get_corpus
from gea.indexes import page_query_vectors, unload_models
from gea.proofs import iou, patches_to_bbox
from gea.retrieval import page_heat, page_size, rank_regions
from gea.safeio import atomic_write_json, read_jsonl

OUT = cfg.PROJECT_ROOT / "results" / "day4_regions.json"
METHODS = ("text", "visual", "both")


def main() -> int:
    questions = read_jsonl(cfg.PROJECT_ROOT / "eval" / "splits" / "dev.jsonl")
    start, rows = time.time(), []
    with cfg.override(BACKEND="real", TRACE=False):
        visual_ok = all(page_query_vectors(q["question"]) is not None for q in questions)
        methods = METHODS if visual_ok else ("text",)
        for q in questions:
            paper = get_corpus(q["paper_id"])
            ref = next(r for r in paper[q["ref_kind"] + "s"] if r["label"] == q["ref_label"])
            row = {"qid": q["qid"], "kind": q["ref_kind"], "ref_label": ref["label"], "ref_page": ref["page"],
                   "candidates": len(paper[q["ref_kind"] + "s"])}
            for how in methods:
                ranked, _, used = rank_regions(paper, q["question"], q["ref_kind"], how)
                order = [r["id"] for r, _ in ranked]
                row[how] = {"rank": order.index(ref["id"]) + 1, "top3": [r["label"] for r, _ in ranked[:3]]}
            if visual_ok:
                heat = page_heat(paper, q["question"])[ref["page"]]
                size = page_size(paper, ref["page"])
                hot = np.unravel_index(int(heat.argmax()), heat.shape)
                rows_, cols_ = heat.shape
                centre = ((hot[1] + 0.5) * size[0] / cols_, (hot[0] + 0.5) * size[1] / rows_)
                row["pointing"] = bool(ref["bbox"][0] <= centre[0] <= ref["bbox"][2]
                                       and ref["bbox"][1] <= centre[1] <= ref["bbox"][3])
                row["map_iou"] = round(iou(patches_to_bbox(heat, page_size=size), tuple(ref["bbox"])), 3)
            rows.append(row)
    unload_models()

    def share(rows_, test):
        return round(sum(test(r) for r in rows_) / len(rows_), 4) if rows_ else None

    summary = {}
    for kind in ("table", "figure", "all"):
        sub = [r for r in rows if kind == "all" or r["kind"] == kind]
        summary[kind] = {"questions": len(sub),
                         "random_hit@1": round(float(np.mean([1 / r["candidates"] for r in sub])), 4)}
        for how in methods:
            summary[kind][f"{how}_hit@1"] = share(sub, lambda r: r[how]["rank"] == 1)
            summary[kind][f"{how}_hit@3"] = share(sub, lambda r: r[how]["rank"] <= 3)
        if visual_ok:
            summary[kind]["pointing"] = share(sub, lambda r: r["pointing"])
            summary[kind]["map_iou_median"] = round(float(np.median([r["map_iou"] for r in sub])), 3)
            summary[kind]["map_iou>=0.5"] = share(sub, lambda r: r["map_iou"] >= 0.5)
    atomic_write_json(OUT, {"split": "dev", "visual_available": visual_ok, "methods": list(methods),
                            "region_search": cfg.REGION_SEARCH, "summary": summary, "questions": rows})

    print(f"{len(rows)} dev questions ({time.time() - start:.0f} s); ColQwen2 heat maps: "
          f"{'yes' if visual_ok else 'NOT AVAILABLE (no room for ColQwen2 and no cached queries)'}\n")
    for kind in ("table", "figure", "all"):
        s = summary[kind]
        parts = [f"{how} {s[f'{how}_hit@1']:.0%}/{s[f'{how}_hit@3']:.0%}" for how in methods]
        print(f"  {kind:6s} ({s['questions']:2d} q): right one 1st / in top 3 -- " + ",  ".join(parts)
              + f"   (random 1st: {s['random_hit@1']:.0%})")
        if visual_ok:
            print(f"         heat map: hottest patch on the reference {s['pointing']:.0%}, "
                  f"box IoU median {s['map_iou_median']:.2f}, IoU >= 0.5 for {s['map_iou>=0.5']:.0%}")
    print(f"saved {OUT.relative_to(cfg.PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
