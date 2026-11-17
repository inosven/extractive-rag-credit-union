"""Step 3: keep only items that check out.

Machine checks (no API), with the harness's own scoring.contains:
  * every must_include token is in the source page as it stood in the snapshot that applies to as_of;
  * every must_not_include token occurs somewhere in the data, and is not in expected_answer or must_include;
  * should_refuse: no must_include, and none of absent_terms occurs on any page of the applicable snapshot;
  * answerable items have 1-3 must_include tokens of at most 40 characters.
Independent model check (API, Batch, Claude Sonnet 5.5, a different model from the generator): it reads the
question and the source page and judges whether the reference answer is correct and complete.

    python tools/evalgen/verify.py --machine-only    # no API
    python tools/evalgen/verify.py                   # machine checks + model check -> PRIVATE/verified.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from common import CHECK_MODEL, PRIVATE, clean, estimate_tokens, guard_budget, jdump, run_batch, snapshot_for, snapshots
import scoring  # noqa: E402  (harness/ is put on sys.path by common)

VERDICT = {"type": "object", "properties": {
    "verdict": {"type": "string", "enum": ["correct", "incorrect", "ambiguous"]},
    "note": {"type": "string"}}, "required": ["verdict", "note"], "additionalProperties": False}

CHECK_SYSTEM = """You audit test items for a credit-union question-answering benchmark. You get a member's \
question, the reference date, the reference answer with the tokens a correct answer must contain and must not \
contain, and the website page the item was written from, as it stood on the reference date. Decide:
- correct: the page supports the reference answer, the must-contain tokens are the right values for exactly what \
was asked (right product, tier, term, date), and none of the must-not-contain tokens is a correct answer;
- incorrect: the reference answer or a token is wrong, stale, the wrong tier, or not supported by the page;
- ambiguous: the question could reasonably mean something else, or several answers are defensible.
For a should_refuse item, "correct" means the page does not state the answer and the question is a natural one.
Keep the note under 25 words."""


def item_snapshot_text(snaps, it):
    sid, taken, pages = snapshot_for(snaps, it.get("as_of"))
    title, md = pages.get(it["url"], ("", ""))
    return sid, title, md


def machine_check(snaps, it, all_text_lower: str) -> list[str]:
    errs = []
    sid, title, md = item_snapshot_text(snaps, it)
    if not md:
        errs.append("source page missing in the applicable snapshot")
    flat = clean(md)
    flat2 = re.sub(r"[|*_#]+", " ", flat)
    must, must_not = it.get("must_include") or [], it.get("must_not_include") or []
    if it["category"] == "should_refuse":
        if must:
            errs.append("should_refuse with must_include")
        _, _, pages = snapshot_for(snaps, it.get("as_of"))
        for term in it.get("absent_terms") or []:
            t = term.lower().strip()
            if len(t) < 4:
                continue
            if any(t in clean(m).lower() for _, m in pages.values()):
                errs.append(f"absent term occurs on the site: {term!r}")
        if not it.get("absent_terms"):
            errs.append("should_refuse without absent_terms")
    else:
        if not 1 <= len(must) <= 3:
            errs.append("must_include count")
        for tok in must:
            if len(tok) > 40:
                errs.append(f"must_include too long: {tok!r}")
            elif not (scoring.contains(flat, tok) or scoring.contains(flat2, tok) or scoring.contains(md, tok)):
                errs.append(f"must_include not on source page: {tok!r}")
    for tok in must_not:
        if scoring.contains(it.get("expected_answer", ""), tok):
            errs.append(f"trap occurs in expected_answer: {tok!r}")
        if any(scoring.contains(m, tok) or tok.lower() in m.lower() for m in must):
            errs.append(f"trap overlaps must_include: {tok!r}")
        if tok.lower() not in all_text_lower and not re.fullmatch(r"[\s$\d.,%]+(apr|apy)?", tok.lower().strip()):
            errs.append(f"trap never occurs in the data: {tok!r}")
    if not it.get("question", "").strip():
        errs.append("empty question")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build the model-check requests and estimate cost")
    args = ap.parse_args()
    snaps = snapshots()
    all_text_lower = "\n".join(clean(md).lower() for _, _, pages in snaps for _, md in pages.values())
    raw = [json.loads(l) for l in (PRIVATE / "raw_items.jsonl").read_text().splitlines() if l.strip()]
    passed, reasons = [], Counter()
    for i, it in enumerate(raw):
        it["raw_id"] = f"r{i:04d}"
        errs = machine_check(snaps, it, all_text_lower)
        if errs:
            for e in errs:
                reasons[e.split(":")[0]] += 1
        else:
            passed.append(it)
    print(f"machine checks: {len(passed)}/{len(raw)} pass; rejections: {dict(reasons)}")
    jdump([it["raw_id"] for it in passed], PRIVATE / "machine_passed.json")
    if args.machine_only:
        return
    reqs = []
    for it in passed:
        sid, title, md = item_snapshot_text(snaps, it)
        body = (f"Question: {it['question']}\nReference date: {it.get('as_of') or 'newest download'} "
                f"(page as of download {sid})\nCategory: {it['category']}\nReference answer: {it['expected_answer']}\n"
                f"Must contain: {it.get('must_include')}\nMust not contain: {it.get('must_not_include')}\n\n"
                f"--- page {it['url']} ({title}) ---\n{clean(md)[:14000]}")
        reqs.append({"custom_id": it["raw_id"], "params": {
            "model": CHECK_MODEL, "max_tokens": 4000, "system": CHECK_SYSTEM,
            "messages": [{"role": "user", "content": body}],
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": VERDICT}}}})
    from llm import cost_usd
    tin = sum(estimate_tokens(r["params"]["system"] + r["params"]["messages"][0]["content"]) for r in reqs)
    est = cost_usd(CHECK_MODEL, tin, 800 * len(reqs)) * 0.5
    print(f"model check: {len(reqs)} requests, ~{tin} input tokens: ~${est:.2f} at batch rate")
    if args.dry_run:
        return
    guard_budget(est)
    out = run_batch(reqs, "evalcheck", PRIVATE / "check_batch.json")
    kept, verdicts = [], Counter()
    for it in passed:
        try:
            v = json.loads(out.get(it["raw_id"]) or "")
        except ValueError:
            v = {"verdict": "error", "note": ""}
        verdicts[v["verdict"]] += 1
        it["check"] = v
        if v["verdict"] == "correct":
            kept.append(it)
    with (PRIVATE / "verified.jsonl").open("w") as fh:
        for it in kept:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"model check verdicts: {dict(verdicts)}; kept {len(kept)} -> {PRIVATE / 'verified.jsonl'}")


if __name__ == "__main__":
    main()
