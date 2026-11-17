"""The optional LLM layer (submission/llm.py) and its hook in h_retrieve, tested without any API call.

A fake transport stands in for the Anthropic client. The answer tests build the full knowledge base from
snapshots/ once for the module, like the other answer tests.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HAVE_SITE = any((ROOT / "snapshots").glob("0*/manifest.json"))     # the website snapshots are not in the repository
needs_site = unittest.skipUnless(HAVE_SITE, "needs the website snapshots (not distributed); see README")
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))

import llm  # noqa: E402
import solution  # noqa: E402
from run_eval import load_snapshots  # noqa: E402

CANDS = [{"id": "p1", "text": "Alpha passage.", "url": "u1"}, {"id": "p2", "text": "Beta passage.", "url": "u2"}]


class FakeTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0) if self.replies else {"ok": True, "text": "pick: none 0 0"}
        if isinstance(r, Exception):
            raise r
        return r


def ok(choice, i=2000, o=5, first=0, last=0):
    """A reply in the three-line format the prompt asks for; only the last line is read."""
    text = f"asks: something\ncheck: {choice} states it\npick: {choice} {first} {last}"
    return {"ok": True, "text": text, "usage": {"input_tokens": i, "output_tokens": o}}


def client(replies, **kw):
    t = FakeTransport(replies)
    return llm.LLMClient("answer", transport=t, **kw), t


class FromEnv(unittest.TestCase):
    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(llm.LLMClient.from_env())

    def test_off_without_key(self):
        with mock.patch.dict(os.environ, {"DUPACO_LLM": "answer"}, clear=True):
            self.assertIsNone(llm.LLMClient.from_env())

    def test_unknown_mode_is_off(self):
        with mock.patch.dict(os.environ, {"DUPACO_LLM": "yes", "ANTHROPIC_API_KEY": "x"}, clear=True):
            self.assertIsNone(llm.LLMClient.from_env())

    def test_key_file(self):
        with tempfile.TemporaryDirectory() as d:
            kf = Path(d) / "key"
            kf.write_text("export ANTHROPIC_API_KEY='sk-test'\n")
            with mock.patch.dict(os.environ, {"DUPACO_LLM": "answer", "DUPACO_LLM_KEY_FILE": str(kf)}, clear=True):
                c = llm.LLMClient.from_env()
                try:
                    import anthropic  # noqa: F401
                    self.assertIsNotNone(c)
                    self.assertEqual(os.environ.get("ANTHROPIC_API_KEY"), "sk-test")
                except ImportError:
                    self.assertIsNone(c)

    def test_solution_has_no_llm_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(solution.build().llm)


class Select(unittest.TestCase):
    def test_valid_choice(self):
        c, t = client([ok("p2")])
        self.assertEqual(c.select_passage("q?", "2026-09-28", "s5", CANDS), "p2")

    def test_reply_formats(self):
        self.assertEqual(llm.parse_reply("p3 2 4"), {"choice": "p3", "first": 2, "last": 4})
        self.assertEqual(llm.parse_reply("asks: a fee\ncheck: p1 states it\npick: p1 0 0")["choice"], "p1")
        self.assertEqual(llm.parse_reply('{"choice": "none", "first": 0, "last": 0}')["choice"], "none")
        for bad in ("", "I think p3 is best", "pick: p3 2", "pick: p3 2 4\nbecause it states the fee"):
            self.assertEqual(llm.parse_reply(bad), {})

    def test_sentence_range(self):
        cands = [dict(CANDS[0], parts=["One.", "Two.", "Three."]), CANDS[1]]
        c, _ = client([ok("p1", first=2, last=3), ok("p1", first=1, last=3), ok("p1", first=3, last=9)])
        self.assertEqual(c.select_passage("q1?", "d", "s", cands), "p1:2-3")
        self.assertEqual(c.select_passage("q2?", "d", "s", cands), "p1")       # the whole passage
        self.assertEqual(c.select_passage("q3?", "d", "s", cands), "p1")       # a range outside the passage is ignored

    def test_wordy_reply_does_not_trip_breaker(self):
        c, _ = client([{"ok": False, "long": True}] * 5)
        for i in range(5):
            self.assertIsNone(c.select_passage(f"q{i}?", "d", "s", CANDS))
        self.assertFalse(c.disabled)

    def test_none(self):
        c, _ = client([ok("none")])
        self.assertEqual(c.select_passage("q?", "d", "s", CANDS), "none")

    def test_invalid_id_falls_back(self):
        c, _ = client([ok("p9")])
        self.assertIsNone(c.select_passage("q?", "d", "s", CANDS))

    def test_malformed_and_error_fall_back(self):
        c, _ = client([{"ok": True, "text": "not json"}, {"ok": False}, RuntimeError("boom")])
        for _ in range(3):
            self.assertIsNone(c.select_passage("q?", "d", "s", CANDS))
        self.assertTrue(c.disabled)          # circuit breaker after 3 failures
        self.assertFalse(c.answer_enabled)

    def test_timeouts_do_not_trip_breaker(self):
        c, _ = client([{"ok": False, "timeout": True}] * 5)
        for i in range(5):
            self.assertIsNone(c.select_passage(f"q{i}?", "d", "s", CANDS))
        self.assertFalse(c.disabled)

    def test_success_resets_breaker(self):
        c, _ = client([{"ok": False}, {"ok": False}, ok("p1"), {"ok": False}])
        for _ in range(4):
            c.select_passage("q?", "d", "s", [dict(x, text=x["text"] + str(_)) for x in CANDS])
        self.assertFalse(c.disabled)

    def test_cache_repeats_without_calling(self):
        c, t = client([ok("p1"), ok("p2")])
        a = c.select_passage("What is X?", "d", "s", CANDS)
        b = c.select_passage("  what is   x? ", "d", "s", CANDS)
        self.assertEqual(a, b)
        self.assertEqual(len(t.calls), 1)

    def test_cache_on_disk(self):
        with tempfile.TemporaryDirectory() as d:
            c, _ = client([ok("p2")], cache_dir=d)
            c.select_passage("q?", "d", "s", CANDS)
            c2, t2 = client([ok("p1")], cache_dir=d)
            self.assertEqual(c2.select_passage("q?", "d", "s", CANDS), "p2")
            self.assertEqual(t2.calls, [])
            self.assertGreater(c2.ledger.total, 0)      # spend ledger reloaded from disk

    def test_no_time_left_no_call(self):
        c, t = client([ok("p1")])
        self.assertIsNone(c.select_passage("q?", "d", "s", CANDS, timeout=llm.MIN_CALL_S - 0.1))
        self.assertEqual(t.calls, [])

    def test_budget_cap(self):
        c, t = client([ok("p1")], budget_usd=0.0)
        self.assertIsNone(c.select_passage("q?", "d", "s", CANDS))
        self.assertEqual(t.calls, [])

    def test_spend_accounting(self):
        c, _ = client([ok("p1", i=1_000_000, o=0)])
        c.select_passage("q?", "d", "s", CANDS)
        self.assertAlmostEqual(c.ledger.total, 0.10, places=6)     # Haiku 5.5 input price per MTok

    def test_prompt_carries_question_date_and_passages(self):
        c, t = client([ok("p1")])
        c.select_passage("When did X open?", "2026-05-12", "s1",
                         [dict(CANDS[0], date="2025-01-02", title="News", section="Openings"), CANDS[1]])
        p = t.calls[0]["prompt"]
        for s in ("Reference date: 2026-05-12", "When did X open?", "[p1]", "dated 2025-01-02", "section: Openings"):
            self.assertIn(s, p)


class Fusion(unittest.TestCase):
    def test_rank_fusion(self):
        import passages
        self.assertEqual(passages.fuse([["a", "b", "c"], ["b", "d"]], 3), ["b", "a", "d"])
        # a low weight lets the third list add candidates without reordering the top of the others
        self.assertEqual(passages.fuse([["a", "b"], ["a", "c"], ["z", "c"]], 4, [1, 1, 0.3]), ["a", "c", "b", "z"])

    def test_retrievers_off_by_default(self):
        import passages
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(passages.Embedder.create())
            self.assertEqual(passages.load_expansions(), {})

    def test_expansions_are_read_from_the_cache_folder(self):
        import passages
        with tempfile.TemporaryDirectory() as d:
            Path(d, "doc2query.jsonl").write_text(json.dumps({"k": "abc", "q": ["How much is a wire?"]}) + "\n")
            with mock.patch.dict(os.environ, {"DUPACO_EXPAND": "on", "DUPACO_LLM_CACHE_DIR": d}):
                self.assertEqual(passages.load_expansions(), {"abc": ["How much is a wire?"]})


S = None
OPEN_Q = "How do I report a lost or stolen debit card?"


def setUpModule():
    global S
    if not HAVE_SITE:
        return
    with mock.patch.dict(os.environ, {}, clear=True):
        S = solution.build()
    for snap in load_snapshots(ROOT / "snapshots"):
        S.ingest(snap)


@needs_site
class Hook(unittest.TestCase):
    """The h_retrieve hook: the model's pick becomes the answer; anything else leaves the rule answer unchanged."""

    def setUp(self):
        S.llm = None
        S._cache.clear()

    def tearDown(self):
        S.llm = None
        S._cache.clear()

    def _rule(self, q):
        S.llm = None
        S._cache.clear()
        a = S.answer(q)
        S._cache.clear()
        return a

    def test_fallback_question_reaches_retrieve(self):
        self._rule(OPEN_Q)
        self.assertEqual(S._last_path, "retrieve")

    def test_failure_keeps_rule_answer(self):
        base = self._rule(OPEN_Q)
        S.llm, _ = client([RuntimeError("down")])
        a = S.answer(OPEN_Q)
        self.assertEqual((a.text, a.confident, [s.url for s in a.sources]),
                         (base.text, base.confident, [s.url for s in base.sources]))
        self.assertEqual(S._last_path, "llm_fail")

    def test_pick_uses_site_text(self):
        S.llm, t = client([ok("p2")])
        a = S.answer(OPEN_Q)
        self.assertEqual(S._last_path, "llm_pick")
        self.assertTrue(a.confident)
        prompt = t.calls[0]["prompt"]
        line = next(x for x in prompt.splitlines() if x.startswith("[p2]"))
        self.assertIn(a.text[:40], line)           # the answer is the chosen passage, not model text
        self.assertTrue(a.sources)

    def test_none_refuses(self):
        S.llm, _ = client([ok("none")] * 6)
        a = S.answer(OPEN_Q)
        self.assertFalse(a.confident)
        self.assertNotRegex(a.text, r"\d")

    RATE_Q = "What's the APR for a 60-month auto loan on a 2025 model-year vehicle?"

    def test_rate_answer_is_reviewed_and_kept(self):
        base = self._rule(self.RATE_Q)
        for reply in (ok("p1"), RuntimeError("down")):
            S._cache.clear()
            S.llm, t = client([reply])
            a = S.answer(self.RATE_Q)
            self.assertEqual(len(t.calls), 1)
            self.assertIn("[p1] ", t.calls[0]["prompt"])
            self.assertIn(base.text[:40], t.calls[0]["prompt"])      # the rule answer is offered as passage 1
            self.assertEqual((a.text, [s.url for s in a.sources]), (base.text, [s.url for s in base.sources]))

    def test_whole_table_question_is_not_reviewed(self):
        S.llm, t = client([ok("p2")])
        S.answer("Do any Dupaco savings tiers pay 0.35% APY?")
        self.assertEqual(t.calls, [])

    def test_amount_question_needs_a_figure(self):
        self.assertFalse(solution.value_stated("how much does a wire cost?", "A fee applies to wire transfers."))
        self.assertTrue(solution.value_stated("how much does a wire cost?", "International wire transfer: $55.00 each."))
        self.assertTrue(solution.value_stated("how do i send a wire?", "Visit any branch."))

    def test_answer_time_budget(self):
        import time as _t

        def slow(**kw):
            _t.sleep(0.5)
            return ok("none")
        S.llm = llm.LLMClient("answer", transport=slow)
        t0 = _t.monotonic()
        S.answer("What did Dupaco announce about its newest learning lab?")
        self.assertLess(_t.monotonic() - t0, solution.LLM_ANSWER_BUDGET_S + 0.6)


@needs_site
class Review(unittest.TestCase):
    """Soft rule handlers: the rule answer is p1; keep on p1/none/failure, swap only to another passage."""
    Q = "How do I update my contact information with Dupaco?"

    @classmethod
    def setUpClass(cls):
        cls.fake = FakeTransport([])
        with mock.patch.object(llm.LLMClient, "from_env",
                               classmethod(lambda c: llm.LLMClient("answer", transport=cls.fake))):
            cls.S = solution.build()
        cls.S.kb = S.kb                      # reuse the module's ingested knowledge base

    def _ask(self, reply):
        self.S._cache.clear()
        self.S.llm = llm.LLMClient("answer", transport=FakeTransport([reply]), use_cache=False)
        return self.S.answer(self.Q)

    def test_keep(self):
        base = self._ask(RuntimeError("down"))
        self.assertEqual(self.S._last_path, "review_fail")
        self.assertEqual(self._ask(ok("p1")).text, base.text)

    def test_none_refuses(self):
        a = self._ask(ok("none"))            # neither the rule answer nor any passage answers the question
        self.assertEqual(self.S._last_path, "review_none")
        self.assertFalse(a.confident)
        self.assertNotRegex(a.text, r"\d")

    def test_swap(self):
        base = self._ask(RuntimeError("down"))
        a = self._ask(ok("p3"))
        self.assertEqual(self.S._last_path, "review_swap")
        self.assertNotEqual(a.text, base.text)
        self.assertTrue(a.confident)


if __name__ == "__main__":
    unittest.main()
