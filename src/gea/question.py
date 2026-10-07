# N1 and N2: understanding the question before any search happens.
import re

from gea import config as cfg
from gea.state import AgentState, FakeTensor
from gea.trace import step


def n1_query_intake(state: AgentState) -> dict:
    # N1 -- Query Intake. Break the question into sub-goals, each needing one kind
    # of evidence.
    #
    # Real version (BUILD_PLAN.md Day 5): one call to the vision-language model,
    # asked for JSON. The stand-in splits on ", and" / ";" and reads the modality
    # off keywords.
    q = state.get("question")
    assert q, "N1: no question in state"
    parts = [p.strip(" ?.") for p in re.split(r";\s*(?:and\s+)?|,\s*and\s+", q) if p.strip(" ?.")]
    goals = []
    for text in parts:
        modality = ("table" if re.search(r"\btables?\b", text, re.I)
                    else "figure" if re.search(r"\b(figures?|plots?|charts?|curves?)\b", text, re.I)
                    else "text")
        goals.append({"text": text[0].upper() + text[1:], "modality": modality})
    listing = "; ".join(f"({i}) {g['modality']}: {g['text']}" for i, g in enumerate(goals, 1))
    return {"sub_goals": goals, "events": step("N1", f"{len(goals)} sub-goal(s): {listing}")}


def n2_query_encoder(state: AgentState) -> dict:
    # N2 -- Query Encoder. Embed the question and pull out the filters every
    # retrieval afterwards depends on: references (Section / Table / Figure N) and
    # names the evidence must mention (a subset, a language).
    #
    # Real version (BUILD_PLAN.md Days 3 and 5): SPECTER2 query embedding
    # (adhoc_query adapter) plus the vision-language model for the filters. The
    # stand-in uses regular expressions.
    q = state.get("question")
    assert q, "N2: no question in state"
    refs = {kind.lower(): num for kind, num in re.findall(r"\b(Section|Table|Figure)\s+(\d+)", q)}
    entities = re.findall(r"([\w-]+)\s+subset", q)
    words = re.findall(r"[A-Za-z][\w-]*", q)
    entities += [w for w in words[1:]
                 if w[0].isupper() and w.lower() not in ("section", "table", "figure")]
    emb = FakeTensor((cfg.SPECTER2_DIM,), "SPECTER2 query")
    return {"query_filters": {"refs": refs, "entities": entities}, "query_embedding": emb,
            "events": step("N2", f"query -> {emb.shape}, refs={refs}, must mention={entities}")}
