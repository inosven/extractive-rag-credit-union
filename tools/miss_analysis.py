"""Why is the answer not among the selector's candidates? Counts only, no question text.

    python tools/miss_analysis.py EVAL.jsonl [--rerank]

For every answerable question whose answer is in none of the 12 candidates (the strict measure of
tools/recall_stats.py: all `must_include` strings in one candidate), the miss is put in the first bucket that
applies:

  1 not on the site      the strings are not all in the text of the snapshot's pages (wrong gold, or the value is
                         only in page metadata such as a publication date)
  2 spread over pages    all strings are on the site, but no single page has them all
  3 spread over a page   one page has them all, but no single passage of the index does (the answer is longer
                         than a 120-word window, or a part was dropped when the page was split)
  4 ranked 13 to 50      a passage has them all and the passage search finds it, below the cut
  5 ranked below 50      a passage has them all; the search does not bring it near the top

With --rerank, a local cross-encoder (fastembed, Xenova/ms-marco-MiniLM-L-6-v2) re-orders the first 50 results
of the fused search and recall@12 is measured again. No model API is called by this script.
"""
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))
from run_eval import load_snapshots  # noqa: E402
from scoring import contains  # noqa: E402

BUCKETS = ["1 not on the site", "2 spread over pages", "3 spread over a page", "4 ranked 13 to 50",
           "5 ranked below 50"]
WIDE = 50


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    spec = importlib.util.spec_from_file_location("solution", ROOT / "submission" / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import llm
    import passages
    ks = mod.build()
    for s in load_snapshots(ROOT / "snapshots"):
        ks.ingest(s)
    items = [json.loads(x) for a in args for x in Path(a).read_text().splitlines() if x.strip()]
    items = [it for it in items if it.get("category") != "should_refuse" and it.get("must_include")]
    ranker = None
    if "--rerank" in sys.argv:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        ranker = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    bucket, by_cat, total_cat = Counter(), defaultdict(Counter), Counter()
    hit12 = rr12 = rr_pool = 0
    rr_ms = []
    for it in items:
        view = ks.kb.view_for(it.get("as_of"))
        info = mod.parse_question(it["question"])
        must = it["must_include"]
        has = lambda t: all(contains(t, m) for m in must)   # noqa: E731
        total_cat[it.get("category", "?")] += 1
        scored = ks.retrieve_scored(view, info)
        cands = [t for _, t in ks._llm_candidates(view, scored, [], info)]
        found = any(has(t) for t in cands)
        hit12 += found
        idx = ks._pidx[view.snapshot_id]
        if ranker is not None:
            llm.MAX_CANDIDATES, keep = WIDE, llm.MAX_CANDIDATES      # the same fusion, 50 results deep
            pool = [t for _, t in ks._llm_candidates(view, scored, [], info)]
            llm.MAX_CANDIDATES = keep
            t0 = time.monotonic()
            scores = list(ranker.rerank(it["question"], [t[:1500] for t in pool]))
            rr_ms.append((time.monotonic() - t0) * 1000)
            order = sorted(range(len(pool)), key=lambda i: -scores[i])
            rr12 += any(has(pool[i]) for i in order[:keep])
            rr_pool += any(has(t) for t in pool)
        if found:
            continue
        pages = {u: p.text for u, p in view.pages.items()}
        if not all(any(contains(t, m) for t in pages.values()) for m in must):
            b = BUCKETS[0]
        elif not any(has(t) for t in pages.values()):
            b = BUCKETS[1]
        elif not any(has(p.text) for p in idx.passages):
            b = BUCKETS[2]
        else:
            top = idx.search(info["toks"], mod.expand_query(info["toks"]), topn=WIDE)
            b = BUCKETS[3] if any(has(p.text) for p in top) else BUCKETS[4]
        bucket[b] += 1
        by_cat[it.get("category", "?")][b] += 1
    n, miss = len(items), sum(bucket.values())
    print(f"{n} answerable questions; answer among the 12 candidates: {hit12} ({hit12 / n:.1%}); missed: {miss}")
    for b in BUCKETS:
        print(f"  {b:22s} {bucket[b]:3d}  {bucket[b] / max(miss, 1):5.0%} of the misses")
    print("misses by category (missed / questions): " + ", ".join(
        f"{c} {sum(v.values())}/{total_cat[c]}" for c, v in sorted(by_cat.items(), key=lambda x: -sum(x[1].values()))))
    if ranker is not None:
        rr_ms.sort()
        print(f"cross-encoder over the first {WIDE} fused results: answer in the pool {rr_pool / n:.3f}; "
              f"recall@12 {hit12 / n:.3f} -> {rr12 / n:.3f}; "
              f"re-rank time p50 {rr_ms[len(rr_ms) // 2]:.0f} ms, p95 {rr_ms[int(len(rr_ms) * 0.95)]:.0f} ms")


if __name__ == "__main__":
    main()
