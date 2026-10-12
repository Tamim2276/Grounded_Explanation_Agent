# Day 6, step 6.3: where does the Faithfulness Gate stop protecting -- and does the model's
# support check (G2 layer 2) move that boundary? The four claim styles of notebook
# Section 15 on the stub paper, written by the template writer, checked by G2 with the
# support check ON (the real model):
#
#   honest         every claim true and cited                 -> should pass both layers
#   fabricated     a number misquoted by +1 (first draft)     -> layer 1 catches it
#   uncited        a number no citation contains              -> layer 1 catches it
#   misattributed  a REAL cell from the WRONG row             -> layer 1 cannot; layer 2?
#
#   bash scripts/start_llm.sh          # first, in its own terminal
#   python scripts/g2_boundary.py      # writes results/day6_g2_boundary.json
import sys

from gea import config as cfg
from gea.generation import draft_claims, fabricate, g2_faithfulness_check
from gea.graph import build_generation_graph
from gea.llm import llm_ready
from gea.safeio import atomic_write_json
from gea.state import new_state
from gea.stub_paper import QUERY

OUT = cfg.PROJECT_ROOT / "results" / "day6_g2_boundary.json"
CASES = {"honest": {}, "fabricated": {"HALLUCINATE_ONCE": True},
         "uncited": {"TABLE_CLAIM_STYLE": "uncited"}, "misattributed": {"TABLE_CLAIM_STYLE": "misattributed"}}


def first_draft_checked(settings: dict, support: str) -> dict:
    # Run the stub graph up to the evidence, write the first draft, and check it once.
    with cfg.override(TRACE=False, **settings):
        state = new_state(QUERY)
        state.update(build_generation_graph().invoke(state))       # to the end, to get the evidence
        draft = draft_claims(state)
        if settings.get("HALLUCINATE_ONCE"):
            draft = fabricate(draft)
        with cfg.override(SUPPORT_CHECK=support):
            checked = g2_faithfulness_check({**state, "claims": draft, "faith_failures": []})
    failures = checked["faith_failures"]
    return {"claims": [c["text"] for c in draft],
            "rejected": [{"claim": f["claim"] + 1, "layer": 2 if f["reason"].startswith("support check") else 1,
                          "reason": f["reason"]} for f in failures]}


def main() -> int:
    if not llm_ready():
        print(f"No model server at {cfg.LLM_URL}: start it first (bash scripts/start_llm.sh)")
        return 1
    results = {}
    for name, settings in CASES.items():
        results[name] = {"layer 1 only": first_draft_checked(settings, "off"),
                         "layers 1 + 2": first_draft_checked(settings, "on")}
        for how, r in results[name].items():
            print(f"{name:14s} {how:13s}: {len(r['claims'])} claim(s), rejected "
                  f"{[(x['claim'], 'L' + str(x['layer'])) for x in r['rejected']] or 'none'}")
            for x in r["rejected"]:
                print(f"{'':30s} claim {x['claim']}: {x['reason'][:110]}")
    atomic_write_json(OUT, {"model": cfg.LLM_MODEL, "cases": results})
    print(f"saved {OUT.relative_to(cfg.PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
