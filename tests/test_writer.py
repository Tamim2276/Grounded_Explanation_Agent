# N9's model writer and G2's support check (BUILD_PLAN.md Day 6) without a model: a fake
# server stands in for llama.cpp. Plus N10b's quote boxes on a real page.
import io
import json

import pytest

import gea.llm as llm
from gea import config as cfg
from gea.generation import as_cite, g2_faithfulness_check, llm_draft, writer_schema
from gea.state import Evidence, new_state

TABLE = "Table 2: Results\nSubset | Gain\nLow-resource | +1.7\nHigh-resource | +4.4"
CELLS = [{"row": "Low-resource", "col": "Gain", "value": "+1.7", "bbox": (0, 0, 1, 1)},
         {"row": "High-resource", "col": "Gain", "value": "+4.4", "bbox": (0, 1, 1, 2)}]


class FakeServer:
    def __init__(self, answer):
        self.answer, self.requests = answer, []

    def __call__(self, request, timeout=None):
        self.requests.append(json.loads(request.data))
        reply = {"choices": [{"message": {"content": json.dumps(self.answer)}}], "usage": {}}
        return io.BytesIO(json.dumps(reply).encode("utf-8"))


@pytest.fixture
def server(monkeypatch, tmp_path):
    def answer_with(answer):
        fake = FakeServer(answer)
        monkeypatch.setattr(llm.urllib.request, "urlopen", fake)
        return fake
    with cfg.override(LLM_CACHE_DIR=tmp_path, TRACE=False):
        yield answer_with


def state_with(claims=None, failures=None, rewrite=0) -> dict:
    table = Evidence("table", "t2", 3, TABLE, (0, 0, 9, 9), "TableRetriever", label="Table 2", cells=CELLS)
    table.id = "E1"
    return {**new_state("Does the gain hold for the low-resource subset?"), "evidence": [table],
            "claims": claims or [], "faith_failures": failures or [], "rewrite_count": rewrite}


def test_the_writer_can_only_cite_items_in_the_buffer():
    schema = writer_schema(["E1", "E2"])
    cite = schema["properties"]["claims"]["items"]["properties"]["cites"]["items"]
    assert all(option["properties"]["evidence"]["enum"] == ["E1", "E2"] for option in cite["anyOf"])


def test_three_cite_kinds_become_the_internal_shape():
    assert as_cite({"evidence": "E1", "row": "Low-resource", "col": "Gain", "value": " +1.7"}) == \
        {"evidence": "E1", "cell": ("Low-resource", "Gain"), "value": "+1.7"}
    assert as_cite({"evidence": "E2", "quote": "the loss flattens "}) == {"evidence": "E2", "quote": "the loss flattens"}
    assert "observation" in as_cite({"evidence": "E3", "observation": "two peaks"})


def test_a_rewrite_tells_the_model_what_failed(server):
    fake = server({"claims": [{"text": "The low-resource gain is +1.7.", "conclusion": True,
                               "cites": [{"evidence": "E1", "row": "Low-resource", "col": "Gain", "value": "+1.7"}]}]})
    bad = [{"text": "The low-resource gain is +4.4.", "cites": []}]
    claims = llm_draft(state_with(bad, [{"claim": 0, "reason": "E1 Low-resource/Gain is +1.7"}], rewrite=1))
    assert claims[0]["cites"][0]["cell"] == ("Low-resource", "Gain")
    asked = fake.requests[0]["messages"][-1]["content"]
    assert "failed the checks" in asked and "E1 Low-resource/Gain is +1.7" in asked


def test_a_rewrite_keeps_the_claims_that_passed(server):
    server({"claims": [{"text": "The low-resource gain is +1.7.", "conclusion": True,
                        "cites": [{"evidence": "E1", "row": "Low-resource", "col": "Gain", "value": "+1.7"}]}]})
    good = {"text": "Table 2 reports gains.", "cites": [{"evidence": "E1", "quote": "Table 2: Results"}]}
    bad = {"text": "The low-resource gain is +4.4.", "cites": []}
    claims = llm_draft(state_with([good, bad], [{"claim": 1, "reason": "wrong row"}], rewrite=1))
    assert claims[0] is good and claims[1]["text"] == "The low-resource gain is +1.7."


def test_each_cite_kind_only_where_it_makes_sense():
    schema = writer_schema(["E1", "E2", "E3"], cell_ids=["E2"], figure_ids=[])
    options = schema["properties"]["claims"]["items"]["properties"]["cites"]["items"]["anyOf"]
    assert [sorted(o["properties"]) for o in options] == [["evidence", "quote"], ["col", "evidence", "row", "value"]]
    assert options[1]["properties"]["evidence"]["enum"] == ["E2"]          # no observation without a figure


MISATTRIBUTED = {"text": "The low-resource gain is +4.4.",            # a real cell, from the wrong row
                 "cites": [{"evidence": "E1", "cell": ("High-resource", "Gain"), "value": "+4.4"}]}


def test_layer_1_passes_the_wrong_row_and_layer_2_catches_it(server):
    with cfg.override(SUPPORT_CHECK="off"):
        assert g2_faithfulness_check(state_with([MISATTRIBUTED]))["faith_failures"] == []
    server({"verdicts": [{"claim": 1, "supported": False, "reason": "+4.4 is the high-resource gain"}]})
    with cfg.override(SUPPORT_CHECK="on"):
        [failure] = g2_faithfulness_check(state_with([MISATTRIBUTED]))["faith_failures"]
    assert failure["reason"] == "support check: +4.4 is the high-resource gain"


def test_a_missing_verdict_counts_as_unsupported(server):
    server({"verdicts": []})
    with cfg.override(SUPPORT_CHECK="on"):
        [failure] = g2_faithfulness_check(state_with([MISATTRIBUTED]))["faith_failures"]
    assert "no verdict" in failure["reason"]


def test_without_a_server_g2_keeps_its_equality_tests():
    with cfg.override(SUPPORT_CHECK="on", LLM_URL="http://localhost:9/v1", LLM_CACHE=False, TRACE=False):
        out = g2_faithfulness_check(state_with([MISATTRIBUTED]))
    assert out["faith_failures"] == []                       # layer 1 alone cannot see the wrong row


def test_a_quote_is_boxed_where_it_is_printed():
    import pymupdf
    from gea.proofs import quote_box
    path = cfg.DATA_DIR / "corpus" / "1803.03467v4" / "corpus.json"
    if not path.exists():
        pytest.skip("1803.03467v4 not read")
    paper = json.loads(path.read_text(encoding="utf-8"))
    figure = next(r for r in paper["figures"] if r["label"] == "Figure 2")
    with pymupdf.open(paper["pdf"]) as doc:
        label = quote_box(doc[figure["page"] - 1], "Ripple Set", tuple(figure["bbox"]))
        sentence = quote_box(doc[5], "We use Microsoft Satori to construct the knowledge graph for each dataset.")
    assert label and label[2] - label[0] < 40                       # one label, not every occurrence
    assert sentence and sentence[1] > 500 and sentence[3] - sentence[1] < 25
