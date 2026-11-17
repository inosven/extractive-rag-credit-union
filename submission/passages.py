"""Passage-level index for the optional LLM layer (never built when the layer is off).

The rule-based fallback searches single sentences, and rate tables are not in that index at all: their cells
become structured `Fact`s that only the rule handlers read. Measured on the generated dev set, the answer was
missing from the model's 12 candidates for about half of the questions, so the selector could not pick it.
This module gives the selector two more kinds of candidate:

* **prose passages**: overlapping windows of about 120 words over a page's sentences, so an answer that spans
  several sentences is one candidate, and the page title and headings count as words of the passage;
* **fact rows**: one line of text per table row, built from the *winning* fact of each row (newest effective
  date, then page authority, as in `solution.pick_rows`). A stale value on an old page never becomes a
  candidate, and a rate question the rule handlers misread can still be answered from the table.

`fuse()` merges the ranked lists of the different retrievers into one candidate list.

Optional third retriever (DUPACO_EMBED=on, needs the `fastembed` package): a small local embedding model
(BAAI/bge-small-en-v1.5, 67 MB, CPU) ranks the same passages by meaning instead of by shared words. It finds
answers worded differently from the question; on its own it is weaker than BM25 on this site, so it gets a
lower weight in the fusion. Passage vectors are computed once and stored in DUPACO_LLM_CACHE_DIR.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from kb import BM25, Fact, Sentence, SnapshotView, tokenize

WINDOW_WORDS = int(os.getenv("DUPACO_PASSAGE_WORDS", "120"))      # size of a prose passage
STRIDE_WORDS = WINDOW_WORDS // 2       # step between passage starts (windows overlap by half)
HEAD_WEIGHT = 2         # the page title and section headings are indexed this many times

METRIC_NAMES = {
    "apy": "APY", "dividend_rate": "dividend rate", "apr": "APR", "rate": "rate", "points": "points",
    "payment": "monthly payment example", "min_open": "minimum to open", "min_earn": "minimum balance to earn",
    "fee": "fee",
}
METRIC_ORDER = ("apy", "dividend_rate", "rate", "apr", "points", "payment", "fee", "min_open", "min_earn")


@dataclass
class Passage:
    kind: str                       # "prose" or "fact"
    text: str                       # what the model reads and what the answer quotes
    anchor: Sentence                # first sentence (prose) or a stand-in sentence on the fact's page (fact)
    section: str = ""
    effective: Optional[str] = None  # fact rows: the effective date printed under the table
    toks: list = field(default_factory=list)
    parts: list = field(default_factory=list)   # prose: the sentences of the window, in order

    @property
    def url(self) -> str:
        return self.anchor.page.url


def _clean_title(title: str) -> str:
    return re.sub(r"\s*\|\s*Dupaco.*$", "", title or "").strip()


def prose_passages(view: SnapshotView, list_after=None) -> list[Passage]:
    """`list_after(page_text, sentence)` returns the bullet items that follow a sentence ending in ':'; the
    sentence index drops short bullets, so they are put back here ("whitelist these domains: a; b; c.")."""
    out: list[Passage] = []
    for url, sents in view.page_sents.items():
        if not sents:
            continue
        title = _clean_title(sents[0].page.title)
        full = []
        for s in sents:
            items = list_after(s.page.text, s.text) if list_after and s.text.rstrip().endswith(":") else None
            full.append(s.text + " " + "; ".join(items) + "." if items else s.text)
        words = [len(t.split()) for t in full]
        i = 0
        while i < len(sents):
            j, n = i, 0
            while j < len(sents) and (n < WINDOW_WORDS or j == i):
                n += words[j]
                j += 1
            part = sents[i:j]
            sections = list(dict.fromkeys(s.section for s in part if s.section))
            texts = full[i:j]
            body = " ".join(texts)
            toks = tokenize(body) + tokenize(title + " " + " ".join(sections)) * HEAD_WEIGHT
            out.append(Passage("prose", body, part[0], sections[0] if sections else "", None, toks, texts))
            if j >= len(sents):
                break
            k, m = i, 0
            while k < j - 1 and m < STRIDE_WORDS:
                m += words[k]
                k += 1
            i = max(k, i + 1)
    return out


def fact_row_text(metrics: dict) -> str:
    """'Regular Certificates, 36 Month: 2.10% APY, 2.09% dividend rate, $1,000.00 minimum balance to earn.'"""
    best: Fact = next(iter(metrics.values()))[0]
    label = re.sub(r"\s*\|\s*", ", ", best.label).strip(" *")
    head = best.section.strip(" -*#")
    name = label if (not head or head.lower() in label.lower()) else f"{head}, {label}"
    parts = []
    for m in METRIC_ORDER:
        if m in metrics:
            f = metrics[m][0]
            v = f.value.strip()
            if not v or v.lower() in ("none", "n/a", "-"):
                continue
            detail = f.attrs.get("detail") if m == "fee" else None
            parts.append(v + (f" {detail}" if detail else "") if m == "fee" else f"{v} {METRIC_NAMES[m]}")
    return f"{name}: {', '.join(parts)}." if parts else ""


def fact_passages(view: SnapshotView, rows: dict) -> list[Passage]:
    """`rows` is solution.pick_rows(view.facts): row id -> {metric: (winning Fact, facts agreeing with it)}."""
    out: list[Passage] = []
    for metrics in rows.values():
        text = fact_row_text(metrics)
        best: Fact = max((x[0] for x in metrics.values()), key=lambda f: (f.effective or "", f.authority))
        page = view.pages.get(best.url)
        if not text or page is None:
            continue
        toks = tokenize(text) + tokenize(_clean_title(page.title))
        anchor = Sentence(text, page, toks, 0, best.section, -1)
        anchor.effective = best.effective      # shown to the model as the passage's date
        out.append(Passage("fact", text, anchor, best.section, best.effective, toks))
    return out


EMBED_MODEL = os.getenv("DUPACO_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # bge models: queries only
EMBED_CHARS = 1500      # the model reads about 512 tokens; longer text is cut


class Embedder:
    """Local embedding model with a disk store of passage vectors (text hash -> unit vector).
    `Embedder.create()` returns None when DUPACO_EMBED is not "on" or fastembed/numpy are not installed."""

    def __init__(self, model, np, cache_dir: Optional[str]):
        self.model, self.np = model, np
        name = "emb_" + re.sub(r"\W+", "_", EMBED_MODEL) + ".npz"
        self.path = os.path.join(os.path.expanduser(cache_dir), name) if cache_dir else None
        self.store: dict = {}
        if self.path and os.path.exists(self.path):
            z = np.load(self.path)
            self.store = dict(zip(z["keys"].tolist(), z["vecs"]))

    @classmethod
    def create(cls) -> Optional["Embedder"]:
        if os.getenv("DUPACO_EMBED", "off").strip().lower() != "on":
            return None
        try:
            import numpy as np
            from fastembed import TextEmbedding
            return cls(TextEmbedding(EMBED_MODEL), np, os.getenv("DUPACO_LLM_CACHE_DIR") or None)
        except Exception:
            return None

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()

    def _unit(self, v):
        return (v / (self.np.linalg.norm(v) or 1.0)).astype("float32")

    def passages(self, texts: list):
        """Matrix of unit vectors, one row per text; new texts are embedded and added to the disk store."""
        np = self.np
        keys = [self._key(t) for t in texts]
        new = {k: t for k, t in zip(keys, texts) if k not in self.store}
        if new:
            for k, v in zip(new, self.model.embed(list(new.values()), batch_size=64)):
                self.store[k] = self._unit(v)
            if self.path:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                ks = list(self.store)
                np.savez(self.path, keys=np.array(ks), vecs=np.stack([self.store[k] for k in ks]))
        return np.stack([self.store[k] for k in keys]) if keys else np.zeros((0, 1), dtype="float32")

    def query(self, question: str):
        return self._unit(next(iter(self.model.embed([EMBED_QUERY_PREFIX + question]))))


def load_expansions() -> dict:
    """passage text hash -> questions the passage answers, written offline by tools/expand_passages.py
    (document expansion). Empty unless DUPACO_EXPAND=on and the file exists in DUPACO_LLM_CACHE_DIR."""
    cdir = os.getenv("DUPACO_LLM_CACHE_DIR")
    if os.getenv("DUPACO_EXPAND", "off").strip().lower() != "on" or not cdir:
        return {}
    out = {}
    try:
        with open(os.path.join(os.path.expanduser(cdir), "doc2query.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                out[rec["k"]] = rec["q"]
    except (OSError, ValueError, KeyError):
        return {}
    return out


class PassageIndex:
    """BM25 over the prose passages and fact rows of one snapshot view; with an `Embedder`, also their vectors."""

    def __init__(self, view: SnapshotView, rows: dict, list_after=None, embedder: Optional[Embedder] = None,
                 expansions: Optional[dict] = None):
        self.passages = prose_passages(view, list_after) + fact_passages(view, rows)
        for p in self.passages if expansions else ():
            qs = expansions.get(hashlib.sha1(p.text.encode()).hexdigest())
            if qs:                  # searchable words only: the passage text shown to the model is unchanged
                p.toks = p.toks + tokenize(" ".join(qs))
        self.bm25 = BM25([p.toks for p in self.passages]) if self.passages else None
        self.embedder = embedder if self.passages else None
        if self.embedder:
            # title and heading first: they say what the passage is about, as in the BM25 index
            self.vecs = embedder.passages([". ".join(x for x in (_clean_title(p.anchor.page.title), p.section, p.text)
                                                     if x)[:EMBED_CHARS] for p in self.passages])

    def search_vec(self, question: str, topn: int = 20) -> list[Passage]:
        """Passages nearest to the question by cosine similarity ([] without an embedder)."""
        if not self.embedder:
            return []
        sims = self.vecs @ self.embedder.query(question)
        return [self.passages[i] for i in self.embedder.np.argsort(-sims)[:topn]]

    def search(self, toks: list, weights: Optional[dict] = None, topn: int = 20) -> list[Passage]:
        if not self.bm25 or not toks:
            return []
        return [self.passages[i] for i, _ in self.bm25.search(toks, topn=topn, weights=weights)]


RRF_K = 60


def fuse(rankings: list[list], limit: int, weights: Optional[list] = None) -> list:
    """Merge several best-first lists of candidates into one list of at most `limit`, without duplicates.

    Each ranking comes from a different retriever (the sentence retriever with its rule-based re-ranking, the
    passage BM25, the embedding search). Their scores are not comparable, so only the positions are used:
    weighted reciprocal rank fusion, score(c) = sum over lists of weight / (RRF_K + rank), ties broken by the
    order of first appearance. A weight below 1 makes a weaker retriever add candidates without pushing the
    stronger retrievers' top results down. Items must be hashable.
    """
    score: dict = {}
    order: dict = {}
    for i, ranking in enumerate(rankings):
        w = weights[i] if weights else 1.0
        for rank, c in enumerate(ranking):
            score[c] = score.get(c, 0.0) + w / (RRF_K + rank + 1)
            order.setdefault(c, len(order))
    return sorted(score, key=lambda c: (-score[c], order[c]))[:limit]
