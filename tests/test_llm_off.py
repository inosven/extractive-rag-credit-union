"""With the LLM layer off (the default), the working tree must answer exactly like the rule-only reference.

The reference is the git tag `rules-baseline` when it exists, otherwise the branch `main`. Its submission/ (taken
with `git show REF:...`) and the working tree's submission/ run in separate processes over the public and dev
eval sets, and every Answer and conflicts() field is compared. Skipped without git, without the reference, or
without unpacked snapshots. Slow (two full ingests); run on its own:

    python -m unittest tests.test_llm_off
"""
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / ".llm_cache" / "off_check"          # git-ignored; the dumps quote site text


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


REF = "rules-baseline" if _git("rev-parse", "--verify", "refs/tags/rules-baseline").returncode == 0 else "main"


@unittest.skipUnless(shutil.which("git") and _git("rev-parse", "--verify", REF).returncode == 0
                     and any((ROOT / "snapshots").glob("0*")), "needs git, a main branch and unpacked snapshots")
class OffEqualsMain(unittest.TestCase):
    def test_answers_identical(self):
        main_sub = TMP / "main" / "submission"
        main_sub.mkdir(parents=True, exist_ok=True)
        files = _git("ls-tree", "--name-only", REF, "submission/").stdout.split()
        for f in files:
            (main_sub / Path(f).name).write_text(_git("show", f"{REF}:{f}").stdout)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("DUPACO_LLM", "ANTHROPIC_"))}
        # main itself orders some source lists by set iteration (e.g. branch-hours sources), which varies with
        # string hash randomisation between processes; pin the seed so only real differences show
        env["PYTHONHASHSEED"] = "0"
        dumps = {}
        for name, sub in (("main", main_sub), ("branch", ROOT / "submission")):
            out = TMP / f"{name}.json"
            r = subprocess.run([sys.executable, str(ROOT / "tools" / "dump_answers.py"), str(sub), str(out)],
                               cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            dumps[name] = json.loads(out.read_text())
        a, b = dumps["main"], dumps["branch"]
        self.assertEqual(sorted(a), sorted(b))
        diff = [k for k in a if a[k] != b[k]]
        self.assertEqual(diff, [], f"{len(diff)} answers differ, first: {diff[:5]}")
        self.assertGreater(len(a), 240)


if __name__ == "__main__":
    unittest.main()
