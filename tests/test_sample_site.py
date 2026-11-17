"""End-to-end run on the small made-up site in sample/ (three snapshots of thirteen pages). Needs no website data,
no API key and no network, so it also runs in CI. The system was not tuned on this site."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))

import solution  # noqa: E402
from run_eval import load_snapshots  # noqa: E402
from scoring import score_item  # noqa: E402

S = None


def setUpModule():
    global S
    S = solution.build()
    for snap in load_snapshots(ROOT / "sample" / "snapshots"):
        S.ingest(snap)


class SampleSite(unittest.TestCase):
    def test_current_value_beats_stale_pages(self):
        a = S.answer("What is the APY on a 12 month certificate?")
        self.assertIn("4.00%", a.text)                       # the newest rate table
        for old in ("3.50%", "3.75%", "4.75%"):              # earlier snapshots and a 2024 blog post
            self.assertNotIn(old, a.text)
        self.assertTrue(any(s.url.endswith("/rates") for s in a.sources))

    def test_as_of_date(self):
        a = S.answer("What was the 12 month certificate APY in February 2026?", "2026-02-15")
        self.assertIn("3.50%", a.text)
        self.assertNotIn("4.00%", a.text)

    def test_refuses_what_the_site_does_not_state(self):
        a = S.answer("What is the rate on a 24 month certificate?")
        self.assertFalse(a.confident)
        self.assertNotRegex(a.text, r"\d\s?%")

    def test_conflict_is_reported_with_the_winner(self):
        cs = [c for c in S.conflicts() if "12 month certificate" in c.topic.lower()]
        self.assertEqual(len(cs), 1)
        self.assertTrue({"3.50%", "4.00%"} <= set(cs[0].values))       # the rate moved between snapshots
        self.assertIn("4.00%", cs[0].resolution or "")

    def test_score_on_the_sample_questions(self):
        items = [json.loads(x) for x in (ROOT / "sample" / "eval.jsonl").read_text().splitlines() if x.strip()]
        right = sum(score_item(it, S.answer(it["question"], it.get("as_of")))["correctness"] >= 0.999 for it in items)
        self.assertGreaterEqual(right, 16, f"{right} of {len(items)} sample questions answered correctly")


if __name__ == "__main__":
    unittest.main()
