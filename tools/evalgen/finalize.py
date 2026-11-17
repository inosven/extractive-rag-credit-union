"""Step 4 (no API): expand paraphrases into groups, drop near-duplicates of existing questions, split by group into
llm_dev (tuning allowed) and llm_test (sealed), and record their sha256.

    python tools/evalgen/finalize.py [--input verified.jsonl] [--seed 20261114]

Output in PRIVATE: llm_dev.jsonl, llm_test.jsonl, SPLIT.md. Same schema as eval/public_eval.jsonl.
Sealing rule for llm_test: read only its aggregate and per-category scores; never its questions or per-question
results while the system is being changed. Run it only at milestones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import date

from common import PRIVATE, ROOT

STOP = set("the a an of to for is are do does what how my i in on at and or with can be it this that dupaco "
           "your you me there any".split())


def words(q: str) -> set:
    return {w for w in re.findall(r"[a-z0-9$%.]+", q.lower()) if w not in STOP}


def jacc(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="verified.jsonl")
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--max-sim", type=float, default=0.6)
    args = ap.parse_args()
    src = [json.loads(l) for l in (PRIVATE / args.input).read_text().splitlines() if l.strip()]
    existing = []
    for f in [ROOT / "eval" / "public_eval.jsonl", *sorted((ROOT / "dev").glob("*.jsonl"))]:
        existing += [words(json.loads(l)["question"]) for l in f.read_text().splitlines() if l.strip()]

    groups, dropped = [], defaultdict(int)
    kept_words: list[set] = []
    for it in src:
        variants = [it["question"]] + [p for p in it.get("paraphrases") or [] if p.strip()]
        variants = [v for v in variants if all(jacc(words(v), e) <= args.max_sim for e in existing)]
        if len(variants) < 1 + len([p for p in it.get("paraphrases") or [] if p.strip()]):
            dropped["near-duplicate of an existing question"] += 1
        if not variants:
            continue
        w0 = words(variants[0])
        if any(jacc(w0, k) > 0.8 for k in kept_words):
            dropped["near-duplicate within the new set"] += 1
            continue
        kept_words.append(w0)
        groups.append((it, variants))

    rnd = random.Random(args.seed)
    by_cat = defaultdict(list)
    for g in groups:
        by_cat[g[0]["category"]].append(g)
    dev, test = [], []
    for cat in sorted(by_cat):
        gs = by_cat[cat]
        rnd.shuffle(gs)
        half = len(gs) // 2
        test += gs[:half]
        dev += gs[half:]

    def rows(gs, prefix):
        out = []
        for gi, (it, variants) in enumerate(gs):
            gid = f"{prefix}-g{gi:03d}" if len(variants) > 1 else None
            for vi, q in enumerate(variants):
                out.append({"id": f"{prefix}-{gi:03d}{'abc'[vi] if len(variants) > 1 else ''}", "question": q,
                            "as_of": it.get("as_of"), "expected_answer": it["expected_answer"],
                            "must_include": it["must_include"], "must_not_include": it["must_not_include"],
                            "category": it["category"], "difficulty": it.get("difficulty", "medium"),
                            "weight": 1.0, "expected_sources": [it["url"]], "group": gid})
        return out

    info = []
    for name, gs, prefix in (("llm_dev", dev, "ld"), ("llm_test", test, "lt")):
        rs = rows(gs, prefix)
        blob = "\n".join(json.dumps(r, ensure_ascii=False) for r in rs) + "\n"
        (PRIVATE / f"{name}.jsonl").write_text(blob)
        cats = defaultdict(int)
        for r in rs:
            cats[r["category"]] += 1
        info.append(f"| {name} | {len(rs)} | {len(gs)} | {sum(1 for r in rs if r['as_of'])} | "
                    f"{hashlib.sha256(blob.encode()).hexdigest()[:16]} | {dict(sorted(cats.items()))} |")
    (PRIVATE / "SPLIT.md").write_text(
        f"# LLM-written eval set, split {date.today().isoformat()} (seed {args.seed})\n\n"
        "| file | items | groups | as_of items | sha256 (first 16) | categories |\n|---|---|---|---|---|---|\n"
        + "\n".join(info) + f"\n\nDropped: {dict(dropped)}\n\n"
        "llm_test is sealed: aggregate and per-category scores only, run at milestones, never used for tuning.\n")
    print((PRIVATE / "SPLIT.md").read_text())


if __name__ == "__main__":
    main()
