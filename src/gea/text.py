# Small text helpers shared by the retrievers, the gates and the generator.
import re
import unicodedata

STOPWORDS = set('''a an and are as at be by did do does for from has have how in is it its
many much of on or our over than that the their this to was we were what when which who
why with'''.split())

NUMBER = re.compile(r"\d+(?:\.\d+)?")
# A number standing on its own in a claim: "6,322,548", "0.920", "+4.4", "61%" -- but not the
# digits inside a name ("MovieLens-1M", "GPT-4", "F1"), which no citation needs to vouch for.
CLAIM_NUMBER = re.compile(r"(?<![\w.,-])\d+(?:[.,]\d+)*(?![\w-])")


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


def names_label(text: str, label: str) -> bool:
    # Does the text name this table or figure? "Table 2" or "Fig. 2" as a whole word,
    # not "Table 21".
    variants = {label, label.replace("Figure", "Fig.")}
    return any(re.search(rf"\b{re.escape(v)}(?![\w])", text, re.IGNORECASE) for v in variants)


QUOTE_MARKS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def loose(text: str) -> str:
    # Text as G2 compares a quote with its evidence (Day 7): the quirks of PDF text
    # extraction undone on BOTH sides -- ligatures ("classiﬁer"), curly quotes, words
    # hyphenated at a line end ("in- ference"; then every hyphen, so "state-of-the- art"
    # still matches), spaces before punctuation ("item v ."), spaced decimals ("0 . 5%")
    # and runs of whitespace. Still a word-for-word comparison: no word may differ.
    t = unicodedata.normalize("NFKC", text).translate(QUOTE_MARKS)
    t = re.sub(r"-\s+", "", t).replace("-", "")
    t = re.sub(r"\s+([.,;:!?%)\]])", r"\1", t)
    t = re.sub(r"([(\[])\s+", r"\1", t)
    t = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", t)
    return " ".join(t.split())

