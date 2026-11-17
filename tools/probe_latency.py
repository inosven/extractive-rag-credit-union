"""Measure answer-time latency and cost of the LLM selector on real fallback questions. CALLS THE API.

    python tools/probe_latency.py --eval dev/open_eval.jsonl --models claude-haiku-5-5 claude-sonnet-5-5 --n 20
    python tools/probe_latency.py ... --dry-run      # no API call: prints how many calls and the estimated cost

Sends retrieved website passages to the Anthropic API. Records every call in the spend ledger
($DUPACO_LLM_CACHE_DIR/spend.jsonl, default .llm_cache/) and stops at DUPACO_LLM_BUDGET_USD.
The key is read from ANTHROPIC_API_KEY or from DUPACO_LLM_KEY_FILE (default: ../.anthropic_key next to the
repository folder); it is never printed.
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))
os.environ.setdefault("DUPACO_LLM_KEY_FILE", str(ROOT.parent / ".anthropic_key"))
os.environ.setdefault("DUPACO_LLM_CACHE_DIR", str(ROOT / ".llm_cache"))
from run_eval import load_snapshots  # noqa: E402
import llm  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=ROOT / "dev" / "open_eval.jsonl")
    ap.add_argument("--models", nargs="+", default=[llm.DEFAULT_MODEL])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=10.0, help="per call; long on purpose, to see the real tail")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.environ.pop("DUPACO_LLM", None)               # build the system rule-only, then attach a probe client
    spec = importlib.util.spec_from_file_location("solution", ROOT / "submission" / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ks = mod.build()
    for s in load_snapshots(ROOT / "snapshots"):
        ks.ingest(s)
    qs = []
    for line in args.eval.read_text().splitlines():
        if line.strip():
            it = json.loads(line)
            ks._cache.clear()
            ks.answer(it["question"], it.get("as_of"))
            if ks._last_path == "retrieve":
                qs.append(it)
    qs = qs[: args.n]
    # prompt size estimate with a recording transport (no network)
    sizes = []

    def record(**kw):
        sizes.append(len(kw["system"]) + len(kw["prompt"]))
        return {"ok": True, "text": '{"choice": "none"}'}
    ks.llm = llm.LLMClient("answer", transport=record, use_cache=False)
    for it in qs:
        ks._cache.clear()
        ks._llm_deadline = time.monotonic() + 60
        ks._answer(it["question"], it.get("as_of"))
    est_in = statistics.mean(sizes) / 3.5 if sizes else 0
    calls = len(sizes)
    print(f"{len(qs)} fallback questions, {calls} selector calls per model, ~{est_in:.0f} input tokens per call")
    for m in args.models:
        print(f"  {m}: estimated ${llm.cost_usd(m, int(est_in * calls), 20 * calls):.3f}")
    if args.dry_run:
        return
    if not (os.getenv("ANTHROPIC_API_KEY") or llm._read_key_file(os.environ["DUPACO_LLM_KEY_FILE"])):
        raise SystemExit("no API key: set ANTHROPIC_API_KEY or create the key file (see tools/README.md)")
    os.environ.setdefault("ANTHROPIC_API_KEY", llm._read_key_file(os.environ["DUPACO_LLM_KEY_FILE"]) or "")
    for m in args.models:
        c = llm.LLMClient("answer", model=m, timeout=args.timeout, cache_dir=os.environ["DUPACO_LLM_CACHE_DIR"],
                          budget_usd=float(os.getenv("DUPACO_LLM_BUDGET_USD", "95")), use_cache=False)
        lat, fails = [], 0
        real = c._anthropic_call

        def timed(**kw):
            t = time.monotonic()
            r = real(**kw)
            lat.append(time.monotonic() - t)
            return r
        c._transport = timed
        spent0 = c.ledger.total
        ks.llm = c
        for it in qs:
            ks._cache.clear()
            ks._llm_deadline = time.monotonic() + 60
            c.fails, c.disabled = 0, False
            ks._answer(it["question"], it.get("as_of"))
            fails += ks._last_path == "llm_fail"
        if c.last_error:
            print(f"  last error: {c.last_error[:200]}")
        if lat:
            print(f"  per call: {' '.join(f'{x:.2f}' for x in lat)}")
            print(f"{m}: {len(lat)} calls, p50 {statistics.median(lat):.2f}s, p95 {pct(lat, 0.95):.2f}s, "
                  f"max {max(lat):.2f}s, failures {fails}, spent ${c.ledger.total - spent0:.4f} "
                  f"(ledger total ${c.ledger.total:.2f})")


if __name__ == "__main__":
    main()
