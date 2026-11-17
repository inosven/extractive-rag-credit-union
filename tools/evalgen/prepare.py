"""Step 1 (no API): choose the source material for the eval generator and write PRIVATE/tasks.json.

* page tasks: pages of the newest snapshot, stratified by page kind (seeded random sample);
* change tasks: table rows whose value differs between snapshots, found by a plain table diff written here
  (it does not use the submission's table parser, so the generator does not inherit its blind spots).

    python tools/evalgen/prepare.py [--pages 110] [--changes 45] [--seed 20261114]
"""
from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict

from common import PRIVATE, clean, jdump, kind, snapshots

QUOTA = {"product": 30, "news": 24, "learn": 10, "other": 16, "fees": 8, "rates": 6, "branch": 8, "calculator": 0}


def tables(md: str):
    """Yield (heading, header_cells, row_cells) for each markdown table row."""
    heading, header, in_table = "", None, False
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("#"):
            heading = s.lstrip("#").strip()
            in_table, header = False, None
            continue
        if s.startswith("|") and s.count("|") >= 2:
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) or not c for c in cells):
                continue
            if not in_table:
                header, in_table = cells, True
                continue
            yield heading, header, cells
        else:
            in_table, header = False, None


def table_changes(snaps):
    """{(url, heading, row label, column): {taken_on: value}} for cells whose value changes across snapshots."""
    hist = defaultdict(dict)
    for sid, taken, pages in snaps:
        for url, (title, md) in pages.items():
            for heading, header, cells in tables(md):
                if not cells or not header:
                    continue
                label = re.sub(r"\s+", " ", cells[0])
                for col, val in zip(header[1:], cells[1:]):
                    if re.search(r"\d", val) and re.search(r"%|\$", val):
                        hist[(url, heading, label, col.strip())][taken] = val.strip()
    out = {}
    for k, by_date in hist.items():
        if len(set(by_date.values())) > 1:
            out[k] = by_date
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=110)
    ap.add_argument("--changes", type=int, default=45)
    ap.add_argument("--seed", type=int, default=20261114)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    snaps = snapshots()
    latest = snaps[-1][2]

    by_kind = defaultdict(list)
    for url, (title, md) in sorted(latest.items()):
        txt = clean(md)
        if len(txt) < 900:
            continue
        by_kind[kind(url)].append(url)
    scale = args.pages / sum(QUOTA.values())
    want = {k: min(len(by_kind.get(k, [])), round(q * scale)) for k, q in QUOTA.items()}
    spare = args.pages - sum(want.values())
    for k in ("product", "news", "other", "learn"):        # quota a kind cannot fill goes to the big kinds
        add = min(spare, len(by_kind.get(k, [])) - want[k])
        want[k] += max(0, add)
        spare -= max(0, add)
    page_tasks = []
    for k in sorted(want):
        for url in rnd.sample(by_kind.get(k, []), want[k]):
            page_tasks.append({"type": "page", "url": url, "kind": k})

    changes = table_changes(snaps)
    # one row per (url, heading, label): prefer the latest snapshot's pages, spread over pages and headings
    seen, rows = set(), []
    keys = sorted(changes)
    rnd.shuffle(keys)
    rate_col = re.compile(r"apy|apr|yield|dividend|\brate", re.I)
    keys.sort(key=lambda k: 0 if rate_col.search(k[3]) else 1)          # rate columns first, stable shuffle
    per_url = defaultdict(int)
    for key in keys:
        url, heading, label, col = key
        if (url, heading, label) in seen or per_url[url] >= 4 or url not in latest or kind(url) == "calculator":
            continue
        seen.add((url, heading, label))
        per_url[url] += 1
        rows.append({"type": "change", "url": url, "heading": heading, "label": label, "column": col,
                     "history": changes[key]})
        if len(rows) >= args.changes:
            break
    tasks = page_tasks + rows
    for i, t in enumerate(tasks):
        t["task_id"] = f"t{i:03d}"
    jdump({"snapshots": [(s[0], s[1]) for s in snaps], "tasks": tasks}, PRIVATE / "tasks.json")
    kinds = defaultdict(int)
    for t in page_tasks:
        kinds[t["kind"]] += 1
    print(f"{len(page_tasks)} page tasks {dict(kinds)}; {len(rows)} change tasks "
          f"(of {len(changes)} changing table cells) -> {PRIVATE / 'tasks.json'}")


if __name__ == "__main__":
    main()
