"""Step 2 (API, Batch): Claude Opus 5.5 writes eval items from the tasks in PRIVATE/tasks.json.

    python tools/evalgen/gen.py --dry-run     # build the requests, print the cost estimate, no API call
    python tools/evalgen/gen.py               # submit (or resume) the batch, write PRIVATE/raw_items.jsonl

The generator sees only website pages (and, for change tasks, a table row's history). It never sees the
system under test, its notes, or any existing question set.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from datetime import date, timedelta

from common import (GEN_MODEL, PRIVATE, clean, estimate_tokens, guard_budget, jdump, jload, run_batch,
                    snapshots)

CATEGORIES = ["current_value", "fees", "tier_disambiguation", "how_to", "program_info", "entity_attribution",
              "branch_info", "company_facts", "stale_trap", "temporal_as_of", "should_refuse"]

ITEM = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "expected_answer": {"type": "string"},
        "must_include": {"type": "array", "items": {"type": "string"}},
        "must_not_include": {"type": "array", "items": {"type": "string"}},
        "category": {"type": "string", "enum": CATEGORIES},
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        "paraphrases": {"type": "array", "items": {"type": "string"}},
        "absent_terms": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["question", "expected_answer", "must_include", "must_not_include", "category", "difficulty",
                 "paraphrases", "absent_terms"],
    "additionalProperties": False,
}
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": ITEM}},
          "required": ["items"], "additionalProperties": False}

SYSTEM = """You write test questions for an assistant that answers credit-union members from the credit union's \
own website (Dupaco Credit Union). The website was downloaded five times; the assistant must answer with the value \
that is current on the reference date, the right tier, and must say "I don't know" when the website does not say.

Each item you write is scored automatically:
- must_include: 1 to 3 short tokens (at most 40 characters each) that a correct answer has to contain. Copy them \
exactly from the source text: a number with its unit ("$25", "3.50%", "8 a.m."), a name, or a short phrase. Numbers \
are compared by value, other text by substring, so pick tokens that are specific and unambiguous.
- must_not_include: 0 to 3 values that would make the answer wrong if they appeared: another tier's rate, a \
different fee, an older figure, a near-miss name. Each must really occur on the website (on this page or the \
row history you are given). Never list a value that a correct answer would also contain.
- should_refuse items: the website does not state the answer. must_include is empty, must_not_include lists \
plausible guesses, and absent_terms lists 1 to 3 distinctive phrases that would have to appear on the website if \
it did state the answer (so a search can confirm they are absent).

Write questions the way a member would type them: plain, specific, no URLs, no "this page" or "the article". A \
question must be answerable from the whole website without seeing the page, so name the product, branch, program \
or event precisely (with its year when the site has several similar ones). expected_answer is one or two \
sentences, at most 40 words, in your own words (do not copy more than 15 consecutive words). paraphrases are 0 to \
2 rewordings of the same question with different vocabulary and the same answer."""


def page_prompt(task, title, text, snap_dates):
    return (f"Reference date: the newest download ({snap_dates[-1]}). Downloads: {', '.join(snap_dates)}.\n"
            f"Page: {task['url']}\nTitle: {title}\nPage kind: {task['kind']}\n\n"
            f"Write 3 items from this page: 2 answerable questions about specific facts the page states (prefer "
            f"amounts, rates, dates, names, eligibility, steps; for a news release, the facts of that event), "
            f"and 1 should_refuse item on the same topic asking for a specific detail the page does not give. "
            f"Give at least one item 1 or 2 paraphrases.\n\n--- page text ---\n{text}")


def change_prompt(task, as_of, old_val, cur_val, cur_text, old_text, snap_dates):
    hist = "; ".join(f"{d}: {v}" for d, v in sorted(task["history"].items()))
    return (f"Downloads: {', '.join(snap_dates)}. A question with a reference date is answered from the newest "
            f"download taken on or before that date.\n"
            f"Page: {task['url']}\nTable section: {task['heading']}\nRow: {task['label']}\nColumn: {task['column']}\n"
            f"Value in each download: {hist}\n\n"
            f"Write 2 items about this row. Item 1: the current value (reference date = newest download); its "
            f"must_include is the current value {cur_val!r} written the way the page writes it, and its "
            f"must_not_include holds the older values. Use category current_value or stale_trap. Item 2: the value "
            f"as of {as_of}, which is {old_val!r}; must_include that value, must_not_include the current value. Use "
            f"category temporal_as_of; the question may mention the month but the reference date is passed "
            f"separately. Give item 1 one or two paraphrases. Name the product, term and tier precisely.\n\n"
            f"--- current section of the page ---\n{cur_text}\n\n--- same section in the older download ---\n{old_text}")


def section(md: str, heading: str, width: int = 3500) -> str:
    t = clean(md)
    i = t.find(heading) if heading else -1
    return t[max(0, i - 300): i + width] if i >= 0 else t[:width]


def build_requests(seed: int = 20261114):
    rnd = random.Random(seed)
    data = jload(PRIVATE / "tasks.json")
    snaps = snapshots()
    dates = [s[1] for s in snaps]
    latest = snaps[-1][2]
    reqs, meta = [], {}
    for t in data["tasks"]:
        if t["type"] == "page":
            title, md = latest[t["url"]]
            text = clean(md)[:14000]
            prompt = page_prompt(t, title, text, dates)
            meta[t["task_id"]] = {"type": "page", "url": t["url"], "as_of_item2": None}
        else:
            hist = t["history"]
            cur_val = hist.get(dates[-1])
            if cur_val is None:
                continue
            olds = [i for i, d in enumerate(dates[:-1]) if d in hist and hist[d] != cur_val]
            if not olds:
                continue
            k = rnd.choice(olds)
            lo = date.fromisoformat(dates[k])
            hi = date.fromisoformat(dates[k + 1]) - timedelta(days=1)
            as_of = (lo + timedelta(days=rnd.randint(0, max(0, (hi - lo).days)))).isoformat()
            old_val = hist[dates[k]]
            old_md = snaps[k][2].get(t["url"], ("", ""))[1]
            prompt = change_prompt(t, as_of, old_val, cur_val, section(latest[t["url"]][1], t["heading"]),
                                   section(old_md, t["heading"]), dates)
            meta[t["task_id"]] = {"type": "change", "url": t["url"], "as_of_item2": as_of,
                                  "old_value": old_val, "cur_value": cur_val}
        reqs.append({"custom_id": t["task_id"], "params": {
            "model": GEN_MODEL, "max_tokens": 16000, "system": SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        }})
    return reqs, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    reqs, meta = build_requests()
    tin = sum(estimate_tokens(r["params"]["system"] + r["params"]["messages"][0]["content"]) for r in reqs)
    tout = 5000 * len(reqs)                      # thinking + JSON, a generous guess at effort medium
    from llm import cost_usd
    est = cost_usd(GEN_MODEL, tin, tout) * 0.5
    print(f"{len(reqs)} requests, ~{tin} input tokens, ~{tout} output tokens (guess): ~${est:.2f} at batch rate")
    if args.dry_run:
        return
    guard_budget(est)
    jdump(meta, PRIVATE / "gen_meta.json")
    out = run_batch(reqs, "evalgen", PRIVATE / "gen_batch.json")
    n_items, bad = 0, []
    with (PRIVATE / "raw_items.jsonl").open("w") as fh:
        for tid, text in sorted(out.items()):
            try:
                items = json.loads(text or "")["items"]
            except (ValueError, KeyError, TypeError):
                bad.append(tid)
                continue
            for j, it in enumerate(items):
                m = meta[tid]
                it["task_id"], it["url"], it["task_type"] = tid, m["url"], m["type"]
                it["as_of"] = m["as_of_item2"] if (m["type"] == "change" and j == 1) else None
                if m["type"] == "change":
                    it["row_values"] = {"old": m["old_value"], "current": m["cur_value"]}
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
                n_items += 1
    print(f"{n_items} raw items from {len(out) - len(bad)} tasks; failed tasks: {bad}")


if __name__ == "__main__":
    main()
