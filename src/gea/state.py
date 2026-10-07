# The shared state of the query-time graph, and the one format evidence travels in.
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, TypedDict


@dataclass(frozen=True)
class FakeTensor:
    # Stand-in for a torch tensor: carries the real shape, no data.
    shape: tuple
    note:  str = ""

    def __repr__(self) -> str:
        return f"<FakeTensor {self.shape} {self.note}>"


class AgentState(TypedDict, total=False):
    # Shared state for the grounded-explanation graph (thesis Chapter 4).
    #
    # total=False because nodes contribute keys progressively -- at START only the
    # question and the paper id exist, and asserting otherwise would make every
    # node responsible for fields it knows nothing about.

    # inputs
    question: str
    paper_id: str

    # N1/N2: understanding the question
    sub_goals:       list            # N1 -> [{"text": ..., "modality": "text"|"table"|"figure"}]
    query_filters:   dict            # N2 -> {"refs": {"section": "4"}, "entities": ["low-resource"]}
    query_embedding: Any             # N2 -> SPECTER2 query vector (768,)

    # N5-N7: planning and retrieval
    action:       dict               # N5 -> {"tool", "query", "sub_goal", "planner"}
    tried:        list               # N5 -> (sub_goal, tool) pairs already used
    new_evidence: list               # N6a-N7 -> found this round, not yet in the buffer

    # N8: the evidence buffer -- the ONLY thing N9 can read
    evidence: list                   # plain key: N8 is its only writer
    links:    list                   # N8 -> [(id_a, id_b, reason)]

    # control
    round:         int               # retrieval round, owned by replan_prep
    rewrite_count: int               # regenerate loop counter, owned by rewrite_prep
    status:        str               # ok | insufficient_evidence | unsupported_claims_dropped
    sufficiency:   dict              # G1 -> {"sufficient", "uncovered", "missing"}

    # N9 / G2
    claims:         list             # N9 -> [{"text", "cites": [...]}], replaced every draft
    drafts:         int              # N9 -> drafts written so far
    # plain list, NOT a reducer: only G2 writes it, and with operator.add the
    # failures of an earlier draft would survive a successful rewrite.
    faith_failures: list

    # N10a-N12: the three proofs and the output
    claim_links: list                # N10a
    boxes:       list                # N10b
    dag_trace:   dict                # N10c
    report:      str                 # N11
    answer:      str                 # N12

    # every node appends one entry; N10a-N10c write in the same step -> reducer
    events: Annotated[list, operator.add]


def new_state(question: str, paper_id: str = "demo-2026-001") -> AgentState:
    # Build a valid initial state. Collections start empty rather than absent, and
    # the round counter starts at 1: the first retrieval round needs no permission.
    return {
        "question": question,
        "paper_id": paper_id,
        "round": 1,
        "rewrite_count": 0,
        "drafts": 0,
        "status": "ok",
        "tried": [],
        "new_evidence": [],
        "evidence": [],
        "links": [],
        "claims": [],
        "faith_failures": [],
        "events": [],
    }


def describe_state(state: dict, title: str = "STATE") -> None:
    # Compact state dump -- long lists and records are summarised, not printed.
    print(f"--- {title} " + "-" * max(0, 56 - len(title)))
    for k, v in state.items():
        if isinstance(v, (list, tuple)):
            shown = f"[{len(v)} item(s)]"
        elif isinstance(v, dict):
            shown = "{" + ", ".join(list(v)[:4]) + ("..." if len(v) > 4 else "") + "}"
        elif isinstance(v, str) and len(v) > 48:
            shown = v[:45] + "..."
        else:
            shown = v
        print(f"  {k:16s} {shown}")


@dataclass
class Evidence:
    # One item in the evidence buffer -- the single shared format of N8.
    kind:    str                   # "text" | "table" | "figure"
    source:  str                   # record id in the corpus: "c4", "t3", "f2"
    page:    int
    content: str                   # what the generator reads
    bbox:    tuple                 # (x_min, y_min, x_max, y_max), PDF points
    tool:    str                   # which tool produced it
    score:   float = 0.0
    query:   str = ""              # what it was retrieved for
    label:   str = ""              # "Table 3", "Figure 2"; empty for text
    section: Optional[str] = None
    id:      str = ""              # "E1", "E2", ... assigned by N8
    round:   int = 0               # retrieval round that found it
    cells:      list = field(default_factory=list, repr=False)   # tables: every cell + bbox
    similarity: Any = field(default=None, repr=False)            # figures: patch map (N6c)
    crop:       Any = field(default=None, repr=False)            # zoomed view (N7)

    @property
    def key(self) -> tuple:
        # Two items with the same key are the same evidence.
        return (self.kind, self.source, self.tool)
