"""Run the harness on several eval sets in one mode and print a summary table (aggregates only).

    python tools/milestone.py --label rules                     # DUPACO_LLM unset: rule-only, no API
    python tools/milestone.py --label haiku --llm answer        # LLM layer on (API)
    python tools/milestone.py --label rules --judge             # also the harness's LLM judge (API, Opus 5.5)
    python tools/milestone.py --label no-passages --llm answer --sets llm_dev --set-env DUPACO_PASSAGES=off

Eval sets: public, open, blind, blind2, blind3 and, if present in ../llm_private/evalset/, llm_dev and llm_test.
Results go to .llm_cache/runs/<label>/<set>/ (git-ignored). For llm_test only aggregate and per-category scores
are printed, never questions (it is sealed). With --judge an estimated judge spend is added to the ledger
($0.012 per judged item), because the harness's judge does not report usage.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path(os.getenv("DUPACO_EVAL_PRIVATE", str(ROOT.parent / "llm_private" / "evalset")))
CACHE = Path(os.getenv("DUPACO_LLM_CACHE_DIR", str(ROOT / ".llm_cache")))
JUDGE_USD_PER_ITEM = 0.012

SETS = {"public": ROOT / "eval" / "public_eval.jsonl", "open": ROOT / "dev" / "open_eval.jsonl",
        "blind": ROOT / "dev" / "blind_eval.jsonl", "blind2": ROOT / "dev" / "blind2_eval.jsonl",
        "blind3": ROOT / "dev" / "blind3_eval.jsonl",
        "llm_dev": PRIVATE / "llm_dev.jsonl", "llm_test": PRIVATE / "llm_test.jsonl",
        "control": PRIVATE / "control50.jsonl", "sample": ROOT / "sample" / "eval.jsonl"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--llm", default="off", choices=["off", "answer"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--set-env", nargs="*", default=[], metavar="KEY=VALUE",
                    help="switches for ablation runs, e.g. DUPACO_PASSAGES=off DUPACO_LLM_REVIEW=soft DUPACO_LLM_REASON=off")
    args = ap.parse_args()
    env = {k: v for k, v in os.environ.items() if not k.startswith("DUPACO_LLM")}
    env["PYTHONHASHSEED"] = "0"
    env["DUPACO_LLM_CACHE_DIR"] = str(CACHE)
    if args.llm != "off":
        env["DUPACO_LLM"] = args.llm
        env.setdefault("DUPACO_LLM_KEY_FILE", str(ROOT.parent / ".anthropic_key"))
        if args.model:
            env["DUPACO_LLM_MODEL"] = args.model
    for kv in args.set_env:
        k, _, v = kv.partition("=")
        env[k] = v
    if args.judge and not env.get("ANTHROPIC_API_KEY"):
        sys.path.insert(0, str(ROOT / "submission"))
        from llm import _read_key_file
        k = _read_key_file(env.get("DUPACO_LLM_KEY_FILE") or str(ROOT.parent / ".anthropic_key"))
        if k:
            env["ANTHROPIC_API_KEY"] = k
    rows = []
    for name in args.sets:
        ev = SETS[name]
        if not ev.exists():
            continue
        out = CACHE / "runs" / args.label / name
        cmd = [sys.executable, str(ROOT / "harness" / "run_eval.py"), "--submission", str(ROOT / "submission"),
               "--eval", str(ev), "--out", str(out)]
        if args.judge:
            cmd.append("--judge")
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{name}: harness failed\n{r.stderr[-1500:]}")
            continue
        rep = json.loads((out / "report.json").read_text())
        n = sum(1 for l in ev.read_text().splitlines() if l.strip())
        if args.judge:
            sys.path.insert(0, str(ROOT / "submission"))
            from llm import SpendLedger
            SpendLedger(CACHE / "spend.jsonl", 1e9).add("judge-estimate", "claude-opus-5-5", 0, 0,
                                                         usd=n * JUDGE_USD_PER_ITEM, note=f"{args.label}/{name}")
        cons = rep.get("consistency") or {}
        acc = rep.get("qa_score")
        rows.append((name, n, rep.get("final_score"), acc, cons.get("paraphrase_consistency"),
                     cons.get("repeat_consistency"), (rep.get("speed") or {}).get("p95_s"),
                     rep.get("by_category")))
    print(f"\n## {args.label} (llm={args.llm}{', model=' + args.model if args.model else ''}"
          f"{', judge' if args.judge else ''})\n")
    print("| set | n | final | accuracy | paraphrase | repeat | p95 s |\n|---|---|---|---|---|---|---|")
    for name, n, fs, acc, para, rep_, p95, _ in rows:
        print(f"| {name} | {n} | {fs} | {acc} | {para} | {rep_} | {p95} |")
    for name, *_, cats in rows:
        if name in ("blind3", "llm_test", "control") and isinstance(cats, dict):
            print(f"\n{name} by category: " + ", ".join(
                f"{c} {v.get('score', v) if isinstance(v, dict) else v}" for c, v in sorted(cats.items())))
    summary = CACHE / "runs" / "summary.jsonl"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("a") as fh:
        for name, n, fs, acc, para, rep_, p95, cats in rows:
            fh.write(json.dumps({"label": args.label, "llm": args.llm, "model": args.model, "judge": args.judge,
                                 "set": name, "n": n, "final": fs, "accuracy": acc, "paraphrase": para,
                                 "repeat": rep_, "p95": p95, "by_category": cats}) + "\n")


if __name__ == "__main__":
    main()
