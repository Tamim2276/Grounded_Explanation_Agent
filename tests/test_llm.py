# The model-driven nodes (BUILD_PLAN.md Day 5) without a model: the cache, the
# fallbacks and G1's rule floor. A fake server stands in for llama.cpp.
import io
import json

import pytest

import gea.llm as llm
from gea import config as cfg
from gea.question import n1_query_intake, n2_query_encoder
from gea.state import Evidence, new_state
from gea.sufficiency import g1_sufficiency_check

SCHEMA = {"type": "object", "required": ["tool"], "properties": {"tool": {"type": "string"}}}
ASK = [{"role": "user", "content": "Which tool?"}]


class FakeServer:
    # Answers every request with `answer`, and counts the requests.
    def __init__(self, answer: dict):
        self.answer, self.calls = answer, 0

    def __call__(self, request, timeout=None):
        self.calls += 1
        reply = {"choices": [{"message": {"content": json.dumps(self.answer)}}], "usage": {"total_tokens": 9}}
        return io.BytesIO(json.dumps(reply).encode("utf-8"))


@pytest.fixture
def server(monkeypatch, tmp_path):
    # A fake server and an empty cache folder; returns a function that sets the answer.
    def answer_with(answer: dict) -> FakeServer:
        fake = FakeServer(answer)
        monkeypatch.setattr(llm.urllib.request, "urlopen", fake)
        return fake
    with cfg.override(LLM_CACHE_DIR=tmp_path, TRACE=False):
        yield answer_with


def test_a_reply_is_cached_and_reused(server):
    fake = server({"tool": "table"})
    first = llm.call_llm(ASK, SCHEMA, "plan")
    second = llm.call_llm(ASK, SCHEMA, "plan")
    assert first == second == {"tool": "table"}
    assert fake.calls == 1 and llm.last_call["cached"]
    [entry] = list(cfg.LLM_CACHE_DIR.rglob("*.json"))
    saved = json.loads(entry.read_text(encoding="utf-8"))
    assert saved["messages"] == ASK and saved["answer"] == {"tool": "table"}      # the prompt is kept too


def test_a_new_prompt_version_does_not_reuse_old_replies(server, monkeypatch):
    fake = server({"tool": "table"})
    llm.call_llm(ASK, SCHEMA, "plan")
    monkeypatch.setattr(llm, "PROMPT_VERSION", "v-next")
    llm.call_llm(ASK, SCHEMA, "plan")
    assert fake.calls == 2


def test_a_reply_without_the_required_fields_is_refused_and_not_cached(server):
    server({"something": "else"})
    with pytest.raises(ValueError):
        llm.call_llm(ASK, SCHEMA, "plan")
    assert not list(cfg.LLM_CACHE_DIR.rglob("*.json"))


def test_without_a_server_the_nodes_fall_back_and_say_so():
    q = "Which method has the lowest error, and how does the loss change with depth?"
    with cfg.override(LLM_MODE="llm", LLM_URL="http://localhost:9/v1", LLM_CACHE=False, TRACE=False):
        out = n1_query_intake(new_state(q))
    assert len(out["sub_goals"]) == 2                                   # the regex split
    assert "rules: model unavailable" in out["events"][0]["msg"]


def test_n1_takes_the_models_sub_goals(server):
    server({"sub_goals": [{"text": "Which method has the lowest error", "modality": "table"}]})
    with cfg.override(LLM_MODE="llm"):
        out = n1_query_intake(new_state("Which method has the lowest error?"))
    assert out["sub_goals"] == [{"text": "Which method has the lowest error", "modality": "table"}]
    assert "(model" in out["events"][0]["msg"]


def test_n2_keeps_only_names_that_are_in_the_question(server):
    server({"names": ["MovieLens-1M", "BLEU"]})                         # BLEU is not in the question
    with cfg.override(LLM_MODE="llm"):
        out = n2_query_encoder(new_state("Which model has the best AUC on MovieLens-1M?"))
    assert out["query_filters"]["entities"] == ["MovieLens-1M"]


def judged(state_evidence: list, answer: dict, server) -> dict:
    server(answer)
    state = {**new_state("Which method has the lowest error?"),
             "sub_goals": [{"text": "Which method has the lowest error", "modality": "table"}],
             "query_filters": {"refs": {}, "entities": []}, "evidence": state_evidence}
    with cfg.override(LLM_MODE="llm"):
        return g1_sufficiency_check(state)["sufficiency"]


def text_item(i: int = 1) -> Evidence:
    ev = Evidence("text", f"c{i}", 1, "Our method has the lowest error.", (0, 0, 1, 1), "TextRetriever")
    ev.id = f"E{i}"
    return ev


def test_g1_rule_floor_overrides_a_lenient_judge(server):
    # The model says "sufficient", but no table was ever retrieved for a table sub-goal.
    verdict = judged([text_item()], {"sufficient": True, "uncovered": [], "missing": "",
                                     "answers": [{"sub_question": 1, "evidence": "E1", "fact": "lowest error"}]}, server)
    assert not verdict["sufficient"] and verdict["uncovered"] == [0]
    assert "rule floor" in verdict["missing"]


def test_g1_passes_when_the_judge_and_the_floor_agree(server):
    table = Evidence("table", "t1", 1, "Table 1: Errors\nMethod | Error\nOurs | 1.2", (0, 0, 1, 1),
                     "TableRetriever", label="Table 1")
    table.id = "E2"
    verdict = judged([text_item(), table], {"sufficient": True, "uncovered": [], "missing": "",
                                            "answers": [{"sub_question": 1, "evidence": "E2", "fact": "Ours, 1.2"}]}, server)
    assert verdict["sufficient"] and verdict["answers"][0]["evidence"] == "E2"


def test_g1_needs_the_judge_to_name_a_real_item_and_true_numbers(server):
    table = Evidence("table", "t1", 1, "Table 1: Errors\nMethod | Error\nOurs | 1.2", (0, 0, 1, 1),
                     "TableRetriever", label="Table 1")
    table.id = "E2"
    for answers in ([],                                                              # names nothing
                    [{"sub_question": 1, "evidence": "E7", "fact": "Ours, 1.2"}],     # no such item
                    [{"sub_question": 1, "evidence": "E2", "fact": "Ours, 0.9"}]):    # a number not in E2
        verdict = judged([text_item(), table], {"sufficient": True, "uncovered": [], "missing": "",
                                                "answers": answers}, server)
        assert not verdict["sufficient"] and "named no item" in verdict["missing"]


def test_g1_keeps_the_judges_reason_when_it_says_no(server):
    table = Evidence("table", "t1", 1, "Table 1: Datasets", (0, 0, 1, 1), "TableRetriever", label="Table 1")
    table.id = "E1"
    verdict = judged([table], {"sufficient": False, "uncovered": [1], "missing": "the error values", "answers": []}, server)
    assert not verdict["sufficient"] and verdict["uncovered"] == [0] and verdict["missing"] == "the error values"


def failing_server(monkeypatch, error):
    def fail(request, timeout=None):
        raise error
    monkeypatch.setattr(llm.urllib.request, "urlopen", fail)


@pytest.mark.parametrize("code, counter", [(503, "unreachable"), (429, "unreachable"), (400, "refused")])
def test_an_outage_is_told_apart_from_a_bad_request(monkeypatch, tmp_path, code, counter):
    # Day 7: the runner redoes a question after an outage (503 busy, 429 rate limit), but a
    # refused request is the system's own behaviour (its node falls back to the rules).
    import urllib.error
    failing_server(monkeypatch, urllib.error.HTTPError("u", code, "x", {}, None))
    monkeypatch.setattr(llm, "RETRY_WAITS", (0, 0))           # retried, then given up
    before = dict(llm.stats)
    with cfg.override(LLM_CACHE_DIR=tmp_path), pytest.raises(OSError):
        llm.call_llm(ASK, SCHEMA, "plan")
    assert llm.stats[counter] == before[counter] + 1


def test_no_server_at_all_counts_as_unreachable(monkeypatch, tmp_path):
    failing_server(monkeypatch, ConnectionRefusedError())
    before = llm.stats["unreachable"]
    with cfg.override(LLM_CACHE_DIR=tmp_path), pytest.raises(OSError):
        llm.call_llm(ASK, SCHEMA, "plan")
    assert llm.stats["unreachable"] == before + 1


def test_gemini_gets_the_key_from_env_and_no_seed(monkeypatch, tmp_path):
    sent = []

    def fake(request, timeout=None):
        sent.append((request.full_url, request.get_header("Authorization"), json.loads(request.data)))
        reply = {"choices": [{"message": {"content": json.dumps({"tool": "table"})}}], "usage": {}}
        return io.BytesIO(json.dumps(reply).encode("utf-8"))
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("# secrets\nGEMINI_API_KEY=test-key-123\n", encoding="utf-8")
    with cfg.override(LLM_CACHE_DIR=tmp_path / "cache", PROJECT_ROOT=tmp_path, LLM_PROVIDER="gemini"):
        assert llm.call_llm(ASK, SCHEMA, "plan") == {"tool": "table"}
    url, auth, body = sent[0]
    assert url.startswith(cfg.GEMINI_URL) and auth == "Bearer test-key-123"
    assert "seed" not in body and body["model"] == cfg.GEMINI_MODEL


def test_a_missing_gemini_key_stops_the_run_instead_of_falling_back(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=paste-your-key-here\n", encoding="utf-8")
    with cfg.override(PROJECT_ROOT=tmp_path, LLM_PROVIDER="gemini"), pytest.raises(RuntimeError):
        llm.endpoint()


def test_g1_writes_its_facts_before_its_verdict():
    # llama.cpp fills the fields in schema order: the verdict must come last (dev, Day 7)
    from gea.sufficiency import G1_SCHEMA
    assert list(G1_SCHEMA["properties"])[-1] == "sufficient"


def test_a_busy_reply_is_retried_before_anything_is_counted(monkeypatch, tmp_path):
    # Gemini answers 503 "high demand" often; one retry usually gets through (Day 7)
    import urllib.error
    replies = [urllib.error.HTTPError("u", 503, "busy", {}, None), {"tool": "table"}]

    def flaky(request, timeout=None):
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(r)}}], "usage": {}}).encode())
    monkeypatch.setattr(llm.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(llm, "RETRY_WAITS", (0,))
    before = dict(llm.stats)
    with cfg.override(LLM_CACHE_DIR=tmp_path):
        assert llm.call_llm(ASK, SCHEMA, "plan") == {"tool": "table"}
    assert llm.stats["unreachable"] == before["unreachable"]
