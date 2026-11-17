"""Which answer path each eval question takes, and the score per path (no API call unless DUPACO_LLM is set).

    python tools/path_stats.py RESULTS_DIR EVAL.jsonl [EVAL.jsonl ...]

RESULTS_DIR is a harness --out folder for the same eval file(s) (per_question.jsonl); the score per path is read
from it. Paths: rule (a structured handler), retrieve (generic BM25 fallback), llm_pick / llm_none / llm_fail.
Prints counts only; no question text.
"""
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))
from run_eval import load_snapshots  # noqa: E402


def main() -> None:
    res_dir = Path(sys.argv[1])
    evals = [Path(p) for p in sys.argv[2:]]
    scores = {}
    pq = res_dir / "per_question.jsonl"
    if pq.exists():
        for line in pq.read_text().splitlines():
            r = json.loads(line)
            scores[r["id"]] = (r["score"], r.get("category"))
    spec = importlib.util.spec_from_file_location("solution", ROOT / "submission" / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ks = mod.build()
    for s in load_snapshots(ROOT / "snapshots"):
        ks.ingest(s)
    by = defaultdict(list)
    for ev in evals:
        for line in ev.read_text().splitlines():
            if not line.strip():
                continue
            it = json.loads(line)
            ks._cache.clear()
            ks.answer(it["question"], it.get("as_of"))
            sc = scores.get(it["id"], (None, None))
            by[ks._last_path].append(sc)
    n = sum(len(v) for v in by.values())
    for path, v in sorted(by.items(), key=lambda x: -len(x[1])):
        got = [s for s, _ in v if s is not None]
        mean = f"{sum(got) / len(got):.3f}" if got else "n/a"
        print(f"{path:10s} {len(v):4d} ({len(v) / n:5.1%})  mean score {mean}")
    if ks.llm is not None:
        print(f"model calls {ks.llm.calls}, recorded spend ${ks.llm.ledger.total:.4f}")


if __name__ == "__main__":
    main()
