# N1 and N2: understanding the question before any search happens.
#
# Each has two versions, chosen by cfg.LLM_MODE: the stand-in rules (regular
# expressions) and the local model (BUILD_PLAN.md Day 5) answering in a fixed JSON
# shape. If the model cannot be reached or replies badly, the rules answer instead and
# the trace says so -- the same pattern as N5.
import re

from gea import config as cfg
from gea.state import AgentState, FakeTensor
from gea.trace import step

# --- N1 -----------------------------------------------------------------------------

N1_SYSTEM = (                                   # v2 (notes/prompts.md)
    "You prepare questions about one scientific paper for a search agent. Split the question "
    "into sub-questions only when it clearly asks two or three different things; most questions "
    "stay ONE sub-question. For each, decide where in the paper its answer is most likely PRINTED:\n"
    "- table: numbers that compare methods, models, datasets or settings -- scores, accuracy, "
    "error, counts, statistics, or which one is best, worst, highest or lowest;\n"
    "- figure: anything SHOWN as a picture -- how something changes as another thing varies "
    "(curves, plots, bars), the architecture, framework or pipeline of a method and its parts, "
    "and visual examples or qualitative results;\n"
    "- text: definitions, reasons and explanations that are only written in prose.\n"
    "Examples:\n"
    "'What accuracy does the proposed model reach on ImageNet?' -> table\n"
    "'Which baseline performs worst on the test set?' -> table\n"
    "'How does the validation loss change during training?' -> figure\n"
    "'Which modules make up the proposed network?' -> figure\n"
    "'How do the generated samples look compared with the baselines?' -> figure\n"
    "'Why do the authors normalise the inputs?' -> text\n"
    "Keep the question's own words.")
N1_SCHEMA = {"type": "object", "required": ["sub_goals"],
             "properties": {"sub_goals": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
                 "type": "object", "required": ["text", "modality"],
                 "properties": {"text": {"type": "string"},
                                "modality": {"type": "string", "enum": list(cfg.MODALITIES)}}}}}}


def rule_sub_goals(q: str) -> list:
    # The stand-in: split on ", and" / ";", read the modality off keywords.
    parts = [p.strip(" ?.") for p in re.split(r";\s*(?:and\s+)?|,\s*and\s+", q) if p.strip(" ?.")]
    goals = []
    for text in parts:
        modality = ("table" if re.search(r"\btables?\b", text, re.I)
                    else "figure" if re.search(r"\b(figures?|plots?|charts?|curves?)\b", text, re.I)
                    else "text")
        goals.append({"text": text[0].upper() + text[1:], "modality": modality})
    return goals


def llm_sub_goals(q: str) -> list:
    from gea.llm import call_llm
    answer = call_llm([{"role": "system", "content": N1_SYSTEM},
                       {"role": "user", "content": f"Question: {q}"}], N1_SCHEMA, "n1_sub_goals")
    goals = [{"text": g["text"].strip() or q, "modality": g["modality"]} for g in answer["sub_goals"][:3]]
    if not goals:
        raise ValueError("N1: no sub-goals")
    return goals


def n1_query_intake(state: AgentState) -> dict:
    # N1 -- Query Intake. Break the question into sub-goals, each needing one kind
    # of evidence. The model reads meaning ("Which method has the lowest error?" needs a
    # table); the stand-in only sees the words "table" or "plot".
    q = state.get("question")
    assert q, "N1: no question in state"
    goals, how = model_or_rules(llm_sub_goals, rule_sub_goals, q)
    if cfg.BACKEND == "real" and cfg.KIND_ROUTER == "vote":
        # The splitter's words, the examples' kind: the 7B model mostly answers "text" (8/20
        # right on dev), the vote of labelled SPIQA questions 17/20. Real papers only: the
        # stub's fake paper is not SPIQA.
        from gea.kinds import kind_of
        voted = [kind_of(g["text"]) for g in goals]
        goals = [{**g, "modality": kind} for g, (kind, _, _) in zip(goals, voted)]
        how += "; kinds by vote " + ", ".join(f"{votes}/{k} {kind}" for kind, votes, k in voted)
    if cfg.SUB_GOAL_MODALITY:                         # testing knob (scripts/smoke_real.py)
        goals = [{**g, "modality": cfg.SUB_GOAL_MODALITY} for g in goals]
    listing = "; ".join(f"({i}) {g['modality']}: {g['text']}" for i, g in enumerate(goals, 1))
    return {"sub_goals": goals, "events": step("N1", f"{len(goals)} sub-goal(s){how}: {listing}")}


def model_or_rules(model_fn, rules_fn, *args) -> tuple:
    # (result, how it was made, for the trace): the model when cfg.LLM_MODE is "llm" and it
    # answers properly, otherwise the rules.
    if cfg.LLM_MODE != "llm":
        return rules_fn(*args), ""
    from gea.llm import how_answered
    try:
        return model_fn(*args), how_answered()
    except (OSError, ValueError, KeyError) as e:      # unreachable, or a bad reply
        return rules_fn(*args), f" (rules: model unavailable, {type(e).__name__})"


# --- N2 -----------------------------------------------------------------------------

N2_SYSTEM = (
    "List the specific names a question about a scientific paper is about: datasets, methods or "
    "models, metrics, variables, settings -- exactly as written in the question. Leave out general "
    "words such as 'method', 'model', 'dataset', 'performance' or 'paper'. An empty list is fine.")
N2_SCHEMA = {"type": "object", "required": ["names"],
             "properties": {"names": {"type": "array", "maxItems": 6, "items": {"type": "string"}}}}


def rule_entities(q: str) -> list:
    # The stand-in: words before "subset", and capitalised words after the first.
    entities = re.findall(r"([\w-]+)\s+subset", q)
    words = re.findall(r"[A-Za-z][\w-]*", q)
    return entities + [w for w in words[1:] if w[0].isupper() and w.lower() not in ("section", "table", "figure")]


def llm_entities(q: str) -> list:
    from gea.llm import call_llm
    answer = call_llm([{"role": "system", "content": N2_SYSTEM},
                       {"role": "user", "content": f"Question: {q}"}], N2_SCHEMA, "n2_names")
    return [n.strip() for n in answer["names"] if n.strip() and n.strip().lower() in q.lower()]


def n2_query_encoder(state: AgentState) -> dict:
    # N2 -- Query Encoder. Embed the question and pull out the filters every
    # retrieval afterwards depends on: references (Section / Table / Figure N) and
    # names the evidence must mention.
    #
    # References stay regular expressions on every backend: "Table 2" is read exactly.
    # Names come from the model with cfg.LLM_MODE = "llm" (only names that occur in the
    # question are kept). The embedding is SPECTER2's question vector on the real backend.
    q = state.get("question")
    assert q, "N2: no question in state"
    refs = {kind.lower(): num for kind, num in re.findall(r"\b(Section|Table|Figure)\s+(\d+)", q)}
    entities, how = model_or_rules(llm_entities, rule_entities, q)
    if cfg.BACKEND == "real":
        from gea.indexes import embed_text
        emb = embed_text([q], kind="query")[0]
    else:
        emb = FakeTensor((cfg.SPECTER2_DIM,), "SPECTER2 query")
    return {"query_filters": {"refs": refs, "entities": entities}, "query_embedding": emb,
            "events": step("N2", f"query -> {tuple(emb.shape)}, refs={refs}, must mention={entities}{how}")}
