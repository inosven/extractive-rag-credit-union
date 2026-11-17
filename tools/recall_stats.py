"""Where answers are lost: retrieval or selection. Counts only, no question text.

    python tools/recall_stats.py EVAL.jsonl [--no-answer]

For every question that has an answer on the site (not "should refuse"), a candidate *contains the answer* when
all `must_include` tokens of the item occur in that one candidate (harness `scoring.contains`).

* candidate recall@K: the answer is among the K candidates the selector would be shown, for the candidate builder
  as configured by the environment (DUPACO_PASSAGES=off gives the sentence-only builder);
* sentence recall@50: the answer is among the first 50 results of the sentence retriever;
* end-to-end outcome per question (skipped with --no-answer; makes model calls when DUPACO_LLM is set).

The measure is strict: an answer spread over two candidates counts as not retrieved.
"""
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))
from run_eval import load_snapshots  # noqa: E402
from scoring import contains  # noqa: E402


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_answer = "--no-answer" not in sys.argv
    spec = importlib.util.spec_from_file_location("solution", ROOT / "submission" / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ks = mod.build()
    for s in load_snapshots(ROOT / "snapshots"):
        ks.ingest(s)
    items = [json.loads(x) for a in args for x in Path(a).read_text().splitlines() if x.strip()]
    answerable = [it for it in items if it.get("category") != "should_refuse" and it.get("must_include")]
    ks_k = (3, 6, 12)
    rec = Counter()
    kinds = Counter()
    outcome = Counter()
    by_path = defaultdict(Counter)
    by_handler = defaultdict(Counter)     # rule handler -> was the rule answer right, is the final answer right
    for it in items:
        refuse = it not in answerable
        view = ks.kb.view_for(it.get("as_of"))
        info = mod.parse_question(it["question"])

        def has(t: str) -> bool:
            return all(contains(t, m) for m in it["must_include"])

        in_cands = False
        if not refuse and view is not None:
            scored = ks.retrieve_scored(view, info)
            cands = ks._llm_candidates(view, scored, [], info)
            where = [i for i, (_, t) in enumerate(cands) if has(t)]
            for k in ks_k:
                rec[f"candidate recall@{k}"] += bool(where and where[0] < k)
            in_cands = bool(where)
            if where:
                kinds["fact row" if cands[where[0]][0].idx == -1 else "prose"] += 1
            rec["sentence recall@50"] += any(has(ks._passage(view, st)) for _, st in scored[:50])
        if not do_answer:
            continue
        ks._cache.clear()
        a = ks.answer(it["question"], it.get("as_of"))
        if refuse:
            r = "refuse: correct" if not a.confident else "refuse: answered anyway"
        else:
            trap = any(contains(a.text, m) for m in it.get("must_not_include") or [])
            if has(a.text) and not trap:
                r = "correct"
            elif has(a.text):
                r = "right value plus a trap value"
            elif in_cands:
                r = "in candidates, not chosen" if a.confident else "in candidates, refused"
            else:
                r = "not in candidates"
        outcome[r] += 1
        ra = getattr(ks, "_rule_answer", None)
        if ra is not None and ks._last_path not in ("retrieve", "llm_pick", "llm_none", "llm_fail"):
            if refuse:
                rule_ok, final_ok = not ra.confident, not a.confident
            else:
                good = lambda x: has(x.text) and not any(contains(x.text, m) for m in it.get("must_not_include") or [])
                rule_ok, final_ok = good(ra), good(a)
            h = by_handler[getattr(ks, "_last_handler", None) or "(refusal)"]
            h["n"] += 1; h["rule right"] += rule_ok; h["final right"] += final_ok
            h["fixed"] += (not rule_ok) and final_ok; h["broken"] += rule_ok and not final_ok
        by_path[ks._last_path][r] += 1
    n = len(answerable)
    print(f"{len(items)} questions, {n} with an answer on the site; passages {os.getenv('DUPACO_PASSAGES', 'on')}, "
          f"llm {os.getenv('DUPACO_LLM', 'off')}")
    for k, v in rec.items():
        print(f"  {k:24s} {v / n:.3f}")
    print(f"  first answering candidate: {dict(kinds)}")
    if do_answer:
        print("end-to-end:")
        for k, v in outcome.most_common():
            print(f"  {k:32s} {v:4d}  {v / len(items):6.1%}")
        print("by answer path:")
        for p, c in sorted(by_path.items(), key=lambda x: -sum(x[1].values())):
            print(f"  {p:12s} {dict(c.most_common())}")
        print("rule handlers (answer before and after the model's review):")
        for hname, c in sorted(by_handler.items(), key=lambda x: -x[1]["n"]):
            print(f"  {hname:20s} n {c['n']:3d}  rule right {c['rule right']:3d}  final right {c['final right']:3d}  "
                  f"fixed {c['fixed']:3d}  broken {c['broken']:3d}")
        if ks.llm is not None:
            lat = sorted(ks.llm.latencies)
            pct = (lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]) if lat else (lambda q: 0.0)
            print(f"model calls {ks.llm.calls} (p50 {pct(0.5):.2f} s, p90 {pct(0.9):.2f} s, p95 {pct(0.95):.2f} s), "
                  f"recorded spend ${ks.llm.ledger.total:.4f}, failures {ks.llm.fail_reasons}")


if __name__ == "__main__":
    main()
