# Writing files so that a power cut can never leave a broken one behind.
#
# Load shedding can switch the PC off in the middle of a write. Two rules make that
# harmless:
#   1. A file that is expensive to make (a model reply, an index, a corpus, a PDF) is
#      written under a temporary name first, forced to disk, and only then renamed
#      to its real name. A rename is all-or-nothing, so the real name always holds
#      either the old complete file or the new complete file -- never half of one.
#   2. Results are appended one JSON line at a time and forced to disk. A power cut
#      costs at most the line being written, and read_jsonl skips that broken line.
#
# Files that take seconds to regenerate (a notebook figure, a LaTeX table) do not
# need this; anything that costs GPU time, a download or your own work does.
import json
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(path):
    # Give the block a temporary path next to `path`. If the block finishes, the
    # temporary file is forced to disk and renamed to `path`; if it fails, the
    # temporary file is deleted and `path` is left exactly as it was.
    #
    # The temporary name keeps the extension ("pages.tmp.pt"), because some savers
    # pick the file format from it (PyMuPDF's pix.save, matplotlib's savefig).
    #
    #     with atomic_path(folder / "pages.pt") as tmp:
    #         torch.save(pages, tmp)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        yield tmp
        with open(tmp, "rb+") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():                      # the block failed before the rename
            tmp.unlink()


def atomic_write_text(path, text: str) -> None:
    with atomic_path(path) as tmp:
        Path(tmp).write_text(text, encoding="utf-8")


def atomic_write_json(path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=1))


def append_jsonl(path, record: dict) -> None:
    # Add one record as one line and force it to disk before returning.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _repair_last_line(path)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _repair_last_line(path: Path) -> None:
    # A power cut during append_jsonl can leave the file without its final newline.
    # If that last piece is a complete record, finish the line; if it is half a
    # record, cut it off. Otherwise the next record would be glued onto it and both
    # would be lost.
    with open(path, "rb+") as f:
        data = f.read()
        if not data or data.endswith(b"\n"):
            return
        start = data.rfind(b"\n") + 1
        try:
            json.loads(data[start:])
            f.write(b"\n")                    # complete record: just end the line
        except ValueError:
            f.truncate(start)                 # half a record: drop it


def read_jsonl(path) -> list:
    # Every complete record in the file, in order. A broken LAST line (a power cut
    # during append_jsonl) is skipped; a broken line anywhere else is real damage
    # and raises.
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise ValueError(f"{path}: line {i + 1} is damaged") from None
    return records


def remove_leftovers(folder) -> list:
    # Delete temporary files a power cut left behind (they are never complete).
    # Returns what was removed.
    folder = Path(folder)
    if not folder.exists():
        return []
    leftovers = [p for p in folder.rglob("*")
                 if p.is_file() and (p.stem.endswith(".tmp") or p.suffix == ".tmp")]
    for p in leftovers:
        p.unlink()
    return leftovers
