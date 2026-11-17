"""Document expansion (doc2query) for the passage index: Claude Haiku writes, for every passage, three questions
a member might ask that the passage answers. The questions are added to the passage's words in the BM25 index
(DUPACO_EXPAND=on), so a question worded differently from the page can still find it. Nothing runs at answer time.

    python tools/expand_passages.py --dry-run     # number of passages and cost estimate, no API call
    python tools/expand_passages.py               # submit (or resume) the batch, write CACHE/doc2query.jsonl

One Batch API job at half price. The output quotes website text, so it stays in the cache folder
(DUPACO_LLM_CACHE_DIR, git-ignored). The generator sees only passages: no eval question, no system output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "evalgen"))
from common import estimate_tokens, guard_budget, run_batch  # noqa: E402  (also puts harness/ and submission/ on the path)

import passages  # noqa: E402
import solution  # noqa: E402
from run_eval import load_snapshots  # noqa: E402

MODEL = "claude-haiku-5-5"
CACHE = Path(os.getenv("DUPACO_LLM_CACHE_DIR", str(ROOT / ".llm_cache")))
OUT = CACHE / "doc2query.jsonl"
SYSTEM = ("You get one passage from a credit union's website, with its page title. Write three short questions "
          "that a member might type into a help chat and that this passage answers. Use everyday words that differ "
          "from the passage's wording where a member would say it differently. Each question must be answerable "
          "from the passage alone and must name the product or topic. One question per line, nothing else.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ks = solution.build()
    for snap in load_snapshots(ROOT / "snapshots"):
        ks.ingest(snap)
    done = {json.loads(x)["k"] for x in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = {}
    for view in ks.kb.views:
        idx = passages.PassageIndex(view, solution.pick_rows(view.facts), ks._list_after)
        for p in idx.passages:
            k = hashlib.sha1(p.text.encode()).hexdigest()
            if k not in done and k not in todo:
                todo[k] = f"Page: {passages._clean_title(p.anchor.page.title)}\n" \
                          + (f"Section: {p.section}\n" if p.section else "") + f"Passage: {p.text}"
    tin = sum(estimate_tokens(SYSTEM) + estimate_tokens(t) for t in todo.values())
    est = (tin * 0.10 + len(todo) * 60 * 0.50) / 1e6 * 0.5
    print(f"{len(todo)} passages to expand ({len(done)} done); about {tin} input tokens; estimate ${est:.2f}")
    if args.dry_run or not todo:
        return
    guard_budget(est)
    reqs = [{"custom_id": k, "params": {"model": MODEL, "max_tokens": 120, "system": SYSTEM,
                                        "thinking": {"type": "disabled"}, "output_config": {"effort": "low"},
                                        "messages": [{"role": "user", "content": t}]}} for k, t in todo.items()]
    res = run_batch(reqs, "doc2query", CACHE / "doc2query_batch.json")
    ok = 0
    with OUT.open("a") as fh:
        for k, text in res.items():
            qs = [q.strip(" -*\t") for q in (text or "").splitlines() if q.strip().endswith("?")][:3]
            if qs:
                ok += 1
                fh.write(json.dumps({"k": k, "q": qs}, ensure_ascii=False) + "\n")
    (CACHE / "doc2query_batch.json").unlink()        # the next run starts a new batch for what is still missing
    print(f"wrote questions for {ok} of {len(res)} passages to {OUT}")


if __name__ == "__main__":
    main()
