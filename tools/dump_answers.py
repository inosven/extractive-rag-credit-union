"""Dump every answer and conflicts() of one submission folder as JSON (for regression comparisons).

    python tools/dump_answers.py SUBMISSION_DIR OUT.json [EVAL.jsonl ...]

Defaults to eval/public_eval.jsonl and dev/*.jsonl. The output quotes site text: keep it out of git
(write it under .llm_cache/ or outside the repository).
"""
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from run_eval import load_snapshots  # noqa: E402


def main() -> None:
    sub, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    evals = [Path(p) for p in sys.argv[3:]] or [p for p in (ROOT / "eval" / "public_eval.jsonl", *sorted((ROOT / "dev").glob("*.jsonl"))) if p.exists()]
    sys.path.insert(0, str(sub))
    spec = importlib.util.spec_from_file_location("solution", sub / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ks = mod.build()
    for s in load_snapshots(ROOT / "snapshots"):
        ks.ingest(s)
    res = {}
    for ev in evals:
        for line in ev.read_text().splitlines():
            if line.strip():
                it = json.loads(line)
                res[f"{ev.name}:{it['id']}"] = asdict(ks.answer(it["question"], it.get("as_of")))
    res["__conflicts__"] = [asdict(c) for c in ks.conflicts()]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, indent=1))


if __name__ == "__main__":
    main()
