# A power cut in the middle of a write must never leave a broken file behind.
import json

import pytest

from gea.safeio import (append_jsonl, atomic_path, atomic_write_json, read_jsonl,
                        remove_leftovers)


def test_atomic_write_replaces_the_file_and_leaves_no_temporary(tmp_path):
    target = tmp_path / "corpus.json"
    atomic_write_json(target, {"v": 1})
    atomic_write_json(target, {"v": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["corpus.json"]


def test_a_failed_write_keeps_the_old_file(tmp_path):
    target = tmp_path / "index.json"
    atomic_write_json(target, {"v": 1})
    with pytest.raises(RuntimeError):
        with atomic_path(target) as tmp:
            tmp.write_text('{"v": 2, "half', encoding="utf-8")
            raise RuntimeError("power cut")
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["index.json"]


def test_the_temporary_name_keeps_the_extension(tmp_path):
    with atomic_path(tmp_path / "page_1.png") as tmp:
        assert tmp.name == "page_1.tmp.png"
        tmp.write_bytes(b"png")


def test_append_cuts_off_a_half_written_last_line(tmp_path):
    runs = tmp_path / "full_test.jsonl"
    runs.write_text('{"qid": 1}\n{"qid": 2, "ans', encoding="utf-8")   # power cut mid-line
    append_jsonl(runs, {"qid": 3})
    assert read_jsonl(runs) == [{"qid": 1}, {"qid": 3}]


def test_append_keeps_a_complete_record_that_lost_its_newline(tmp_path):
    runs = tmp_path / "full_test.jsonl"
    runs.write_text('{"qid": 1}\n{"qid": 2}', encoding="utf-8")         # cut before "\n"
    append_jsonl(runs, {"qid": 3})
    assert read_jsonl(runs) == [{"qid": 1}, {"qid": 2}, {"qid": 3}]


def test_read_skips_a_broken_last_line_but_not_a_broken_middle_line(tmp_path):
    ok_end = tmp_path / "a.jsonl"
    ok_end.write_text('{"qid": 1}\n{"qid": 2, "ans', encoding="utf-8")
    assert read_jsonl(ok_end) == [{"qid": 1}]

    damaged = tmp_path / "b.jsonl"
    damaged.write_text('{"qid": 1}\n{"qid": 2, "ans\n{"qid": 3}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_jsonl(damaged)

    assert read_jsonl(tmp_path / "missing.jsonl") == []


def test_remove_leftovers_deletes_only_temporary_files(tmp_path):
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "1611.04684v1.pdf").write_bytes(b"%PDF")
    (tmp_path / "pdfs" / "1611.05742v3.tmp.pdf").write_bytes(b"%PD")
    removed = remove_leftovers(tmp_path)
    assert [p.name for p in removed] == ["1611.05742v3.tmp.pdf"]
    assert [p.name for p in (tmp_path / "pdfs").iterdir()] == ["1611.04684v1.pdf"]
