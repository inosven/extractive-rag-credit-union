"""Remove table footnote digits glued to gold values ("6.49% APR1" -> "6.49% APR", "0.00%1" -> "0.00%").

    python tools/evalgen/fix_footnotes.py ../llm_private/evalset/llm_dev.jsonl [more.jsonl ...]

The generator copied some values from rate tables together with their footnote mark, so a correct answer could
not match them. One fixed pattern is applied to `must_include` and `must_not_include`; nothing else changes.
Only counts are printed (never a question), so the script is safe to run on the sealed test set. The original
file is kept as <name>.orig, and the new sha256 is appended to SPLIT.md next to the file.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

FOOTNOTE = re.compile(r"(\d%|%\s?AP[RY])(\d)(?![\d.,%])")


def fix(path: Path) -> None:
    lines = path.read_text().splitlines()
    out, tokens, items = [], 0, 0
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        changed = False
        for key in ("must_include", "must_not_include"):
            new = [FOOTNOTE.sub(r"\1", t) for t in d.get(key, [])]
            n = sum(a != b for a, b in zip(new, d.get(key, [])))
            if n:
                d[key], tokens, changed = new, tokens + n, True
        items += changed
        out.append(json.dumps(d, ensure_ascii=False) if changed else line)
    if not tokens:
        print(f"{path.name}: nothing to fix")
        return
    orig = path.with_suffix(path.suffix + ".orig")
    if not orig.exists():
        orig.write_text("\n".join(lines) + "\n")
    path.write_text("\n".join(out) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with (path.parent / "SPLIT.md").open("a") as fh:
        fh.write(f"\n- footnote fix: `{path.name}` {items} items, {tokens} tokens changed; original kept as "
                 f"`{orig.name}`; new sha256 `{sha}`\n")
    print(f"{path.name}: {items} items, {tokens} tokens changed; sha256 {sha[:16]}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        fix(Path(p))
