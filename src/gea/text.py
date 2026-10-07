# Small text helpers shared by the retrievers, the gates and the generator.
import re

STOPWORDS = set('''a an and are as at be by did do does for from has have how in is it its
many much of on or our over than that the their this to was we were what when which who
why with'''.split())

NUMBER = re.compile(r"\d+(?:\.\d+)?")


def terms(text: str) -> list:
    # Lower-case content words and numbers; "low-resource" and "3.9" stay whole.
    return [t for t in re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower())
            if t not in STOPWORDS]


def sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def table_text(table: dict) -> str:
    # A table as the text the generator reads: caption, header, one line per row.
    lines = [table["caption"], " | ".join(table["header"])]
    lines += [" | ".join(row) for row in table["rows"]]
    return "\n".join(lines)


def score(query: str, text: str, corpus: dict) -> float:
    # Keyword stand-in for embedding similarity: IDF-weighted word overlap.
    # Deterministic, so every run retrieves the same items.
    idf = corpus["text_index"]["idf"]
    return round(sum(idf.get(t, 0.0) for t in set(terms(query)) & set(terms(text))), 2)
