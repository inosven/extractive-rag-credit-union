"""Optional LLM layer (off by default).

The deterministic system never needs this module. It is used only when the environment variable
DUPACO_LLM is set to "answer" (or "both") AND an Anthropic API key is available AND the `anthropic`
package imports. Otherwise `LLMClient.from_env()` returns None and every code path is the rule-based one.

What the model does here is narrow on purpose: for an open-ended question that reached the BM25 fallback, it
picks which of the retrieved candidate passages answers the question, or says that none does. It never writes
the answer text; the answer is always the website's own passage, so every number in it comes from the site.

Safety rails, all in this file:
* hard per-call timeout and no retries at answer time; any error, refusal or malformed output -> None, and the
  caller falls back to the rule-based choice;
* circuit breaker: after a few consecutive failures the layer switches itself off for the process;
* the reply ends with one short line ("pick: p3 2 4") that is checked in code against the ids and sentence
  counts that were sent; anything else counts as a failure. Before that line the model writes two short lines
  (what is asked, which passage states it), which nobody reads: measured on the dev set it picks better with
  them (thinking is switched off for speed). A JSON-schema output format would give the format guarantee from the API, but it measured about
  0.6 s more time to first token per call, which the 2 s answer limit cannot afford;
* a cache keyed on everything the model sees, so the same question gets the same pick (the current models take
  no temperature parameter, so repeatability comes from this cache);
* a spend ledger with a hard budget: no call is made once the recorded spend reaches the cap.

Environment variables:
  DUPACO_LLM             off | answer | extract | both        (default off)
  DUPACO_LLM_MODEL       model for answer-time selection      (default claude-haiku-5-5)
  DUPACO_LLM_TIMEOUT     seconds per call                     (default 1.8)
  DUPACO_LLM_REASON      on | off: a short written reason before the pick (default on; see SYSTEM_PROMPT)
  DUPACO_LLM_CACHE_DIR   folder for select cache + spend ledger; unset = in-memory only.
                         Keep it out of any public repository: cached entries quote site text.
  DUPACO_LLM_BUDGET_USD  hard cap on recorded spend           (default 95)
  DUPACO_LLM_KEY_FILE    file with a line ANTHROPIC_API_KEY=... read when the variable itself is unset
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

REASON = os.getenv("DUPACO_LLM_REASON", "on").strip().lower() != "off"   # the model writes a short reason before its pick
PROMPT_VERSION = "select-v7" + ("r" if REASON else "")
DEFAULT_MODEL = "claude-haiku-5-5"
MAX_CANDIDATES = int(os.getenv("DUPACO_LLM_CANDIDATES", "12"))
BREAKER_FAILS = 3
MIN_CALL_S = 0.7           # a call with less time left than this would only time out (median call: about 1 s)

# USD per million tokens (input, output), Anthropic first-party list prices; Haiku 5.5 rate for prompts <= 100K.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

SYSTEM_PROMPT = (
    "You help a credit union's member-service assistant answer from the credit union's own website. "
    "You get a member's question and numbered passages copied from the website, listed best-first by a search "
    "engine. Choose the one passage that answers the question, or \"none\" if no passage does.\n"
    "Rules:\n"
    "1. The passage must be about exactly what was asked: the same product, branch, program or event. A passage "
    "about something else does not count, even if it shares words.\n"
    "2. If the question asks for a value (an amount, rate, limit, date, phone number, name or count), the chosen "
    "passage must state that value. A passage that only describes the topic does not answer it.\n"
    "3. Several passages may state the same kind of figure from different dates (news releases from different "
    "years, for example). Choose the one with the latest date; older figures are out of date.\n"
    "4. If two passages answer equally well, choose the one listed first.\n"
    "5. Say \"none\" when no passage states the answer.\n"
    "6. A long passage is split into numbered sentences: <1> ... <2> ... For such a passage also give the shortest "
    "run of consecutive sentences, from `first` to `last`, that answers the question completely and names what it "
    "is about. Leave out sentences about other products, other dates or other amounts. For a passage without "
    "numbered sentences, and for \"none\", give first 0 and last 0.\n"
    + ("Reply in exactly three short lines and nothing else:\n"
       "asks: <what the question asks for, 8 words at most>\n"
       "check: <which passage states it, or why none does, 15 words at most>\n"
       "pick: <passage id> <first> <last>\n"
       "Examples of the last line: \"pick: p3 2 4\", \"pick: p1 0 0\", \"pick: none 0 0\"."
       if REASON else
       "Reply with one line and nothing else, no explanation: the passage id, first and last, separated by spaces. "
       "Examples: \"p3 2 4\", \"p1 0 0\", \"none 0 0\".")
)


_REPLY_RE = re.compile(r"\s*[\"'`]?(p\d{1,2}|none)\b[\"'`]?(?:\s+(\d{1,3})\s+(\d{1,3}))?\s*[.\"'`]?\s*", re.I)


def parse_reply(text: str) -> dict:
    """'p3 2 4' -> {"choice": "p3", "first": 2, "last": 4}; anything that is not exactly one such line -> {}.
    A JSON object with the same three keys is accepted too."""
    text = text.strip()
    if text.startswith("{"):
        try:
            out = json.loads(text)
            return out if isinstance(out, dict) else {}
        except ValueError:
            return {}
    last = text.splitlines()[-1] if text else ""                       # reasoning mode: the pick is the last line
    m = _REPLY_RE.fullmatch(re.sub(r"^\s*pick\s*:\s*", "", last, flags=re.I))
    if not m:
        return {}
    return {"choice": m.group(1).lower(), "first": int(m.group(2) or 0), "last": int(m.group(3) or 0)}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES.get(model, max(PRICES.values()))
    return (input_tokens * pin + output_tokens * pout) / 1e6


def _read_key_file(path: str) -> Optional[str]:
    try:
        for line in Path(path).expanduser().read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                v = line.split("=", 1)[1].strip().strip("'\"")
                return v or None
    except OSError:
        return None
    return None


class SpendLedger:
    """Append-only record of what each call cost; refuses new calls once the cap is reached."""

    def __init__(self, path: Optional[Path], cap_usd: float):
        self.path = path
        self.cap = cap_usd
        self._lock = threading.Lock()
        self.total = 0.0
        if path and path.exists():
            for line in path.read_text().splitlines():
                try:
                    self.total += float(json.loads(line).get("usd", 0.0))
                except (ValueError, AttributeError):
                    pass

    def allow(self, reserve: float = 0.0) -> bool:
        return self.total + reserve < self.cap

    def add(self, purpose: str, model: str, input_tokens: int, output_tokens: int, usd: Optional[float] = None,
            note: str = "") -> float:
        usd = cost_usd(model, input_tokens, output_tokens) if usd is None else usd
        rec = {"t": round(time.time(), 1), "purpose": purpose, "model": model, "in": input_tokens,
               "out": output_tokens, "usd": round(usd, 6)}
        if note:
            rec["note"] = note
        with self._lock:
            self.total += usd
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
        return usd


class SelectCache:
    def __init__(self, path: Optional[Path]):
        self.path = path
        self._lock = threading.Lock()
        self.data: dict[str, Optional[str]] = {}
        if path and path.exists():
            for line in path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                    self.data[rec["k"]] = rec["choice"]
                except (ValueError, KeyError, TypeError):
                    pass

    def get(self, key: str):
        return self.data.get(key, KeyError)

    def put(self, key: str, choice: Optional[str]) -> None:
        with self._lock:
            self.data[key] = choice
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as fh:
                    fh.write(json.dumps({"k": key, "choice": choice}) + "\n")


def cache_key(model: str, question: str, ref_date: str, snapshot_id: str, cands: list[dict], review: bool = False) -> str:
    blob = json.dumps([PROMPT_VERSION + ("/review" if review else ""), model, " ".join(question.lower().split()), ref_date, snapshot_id,
                       [(c["id"], c["text"], c.get("url", ""), c.get("date") or "", len(c.get("parts") or []))
                        for c in cands]],
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


REVIEW_NOTE = (
    "Passage [p1] is the answer the assistant is about to give; it was built from the website's tables and "
    "listings and is usually right. Choose p1 whenever it is about what was asked and states what was asked, even "
    "if another passage says the same in other words or adds detail. Choose another passage only when p1 is about "
    "a different product, branch, program or topic than the question, or does not state what was asked.")


def render_candidates(question: str, ref_date: str, cands: list[dict], review: bool = False) -> str:
    lines = [f"Reference date: {ref_date}", f"Question: {question.strip()}", "", "Passages:"]
    for c in cands:
        meta = [f"page: {c.get('title') or c.get('url', '')}"]
        if c.get("section"):
            meta.append(f"section: {c['section']}")
        if c.get("date"):
            meta.append(f"dated {c['date']}")
        parts = c.get("parts") or []
        body = " ".join(f"<{i}> {s}" for i, s in enumerate(parts, 1)) if len(parts) > 1 else c["text"]
        lines.append(f"[{c['id']}] ({'; '.join(meta)}) {body}")
    lines.append("")
    if review:
        lines.append(REVIEW_NOTE)
    lines.append('Reply with the three lines asks / check / pick.' if REASON
                 else 'Reply with one line only: "<passage id> <first> <last>" or "none 0 0".')
    return "\n".join(lines)


class LLMClient:
    """Thin wrapper around the Anthropic client with the safety rails described in the module docstring."""

    def __init__(self, mode: str, model: str = DEFAULT_MODEL, timeout: float = 1.8,
                 cache_dir: Optional[str] = None, budget_usd: float = 95.0,
                 transport: Optional[Callable[..., dict]] = None, use_cache: bool = True):
        self.mode = mode
        self.model = model
        self.timeout = timeout
        cdir = Path(cache_dir).expanduser() if cache_dir else None
        self.cache = SelectCache(cdir / "select_cache.jsonl" if cdir and use_cache else None)
        if not use_cache:
            self.cache.get = lambda key: KeyError          # measurement runs: always call the model
            self.cache.put = lambda key, choice: None
        self.ledger = SpendLedger(cdir / "spend.jsonl" if cdir else None, budget_usd)
        self.fails = 0
        self.disabled = False
        self.calls = 0
        self.fail_reasons: dict[str, int] = {}
        self.latencies: list[float] = []    # seconds per model call (development statistics only)
        self._transport = transport          # tests inject a fake; real calls go through _anthropic_call
        self._client = None
        self.last_error = ""

    # ---- construction
    @classmethod
    def from_env(cls) -> Optional["LLMClient"]:
        mode = os.getenv("DUPACO_LLM", "off").strip().lower()
        if mode not in ("answer", "extract", "both"):
            return None
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            kf = os.getenv("DUPACO_LLM_KEY_FILE")
            key = _read_key_file(kf) if kf else None
            if not key:
                return None
            os.environ["ANTHROPIC_API_KEY"] = key
        try:
            import anthropic  # noqa: F401
        except Exception:
            return None
        try:
            timeout = float(os.getenv("DUPACO_LLM_TIMEOUT", "1.8"))
            budget = float(os.getenv("DUPACO_LLM_BUDGET_USD", "95"))
        except ValueError:
            return None
        return cls(mode, model=os.getenv("DUPACO_LLM_MODEL", DEFAULT_MODEL), timeout=timeout,
                   cache_dir=os.getenv("DUPACO_LLM_CACHE_DIR") or None, budget_usd=budget,
                   use_cache=os.getenv("DUPACO_LLM_NOCACHE", "") != "1")

    @property
    def answer_enabled(self) -> bool:
        return self.mode in ("answer", "both") and not self.disabled

    # ---- the one answer-time call
    def select_passage(self, question: str, ref_date: str, snapshot_id: str, cands: list[dict],
                       timeout: Optional[float] = None, review: bool = False):
        """Returns a candidate id ("p3"), a candidate id with a sentence range ("p3:2-4", for a passage that has
        `parts`), "none", or None when the layer could not decide (caller keeps its own pick)."""
        if not self.answer_enabled or not cands:
            return None
        cands = cands[:MAX_CANDIDATES]
        key = cache_key(self.model, question, ref_date, snapshot_id, cands, review)
        hit = self.cache.get(key)
        if hit is not KeyError:
            return hit
        t = self.timeout if timeout is None else min(self.timeout, timeout)
        if t < MIN_CALL_S or not self.ledger.allow(reserve=0.01):
            self.fail_reasons["no time left" if t < MIN_CALL_S else "budget"] = self.fail_reasons.get("no time left" if t < MIN_CALL_S else "budget", 0) + 1
            return None
        ids = [c["id"] for c in cands]
        prompt = render_candidates(question, ref_date, cands, review)
        t0 = time.monotonic()
        try:
            self.calls += 1
            res = (self._transport or self._anthropic_call)(model=self.model, system=SYSTEM_PROMPT, prompt=prompt,
                                                            timeout=t)
        except Exception:
            res = None
        self.latencies.append(time.monotonic() - t0)
        if res and res.get("usage"):
            u = res["usage"]
            self.ledger.add("select", self.model, u.get("input_tokens", 0), u.get("output_tokens", 0))
        choice = None
        if res and res.get("ok"):
            out = parse_reply(res.get("text") or "")
            choice = out.get("choice")
            if choice not in ids and choice != "none":
                choice = None
            elif choice != "none":
                n = len(next(c for c in cands if c["id"] == choice).get("parts") or [])
                a, b = out.get("first"), out.get("last")
                if n > 1 and isinstance(a, int) and isinstance(b, int) and 1 <= a <= b <= n and (a, b) != (1, n):
                    choice = f"{choice}:{a}-{b}"
        if choice is None:
            why = "exception" if res is None else ("timeout" if res.get("timeout") else
                                                   ("bad output" if res.get("ok") else (self.last_error or "not ok")[:60]))
            self.fail_reasons[why] = self.fail_reasons.get(why, 0) + 1
            if not (res and (res.get("timeout") or res.get("long"))):   # a slow or wordy reply is not a broken service
                self.fails += 1
                if self.fails >= BREAKER_FAILS:
                    self.disabled = True
            return None
        self.fails = 0
        self.cache.put(key, choice)
        return choice

    def warm_up(self) -> None:
        """One tiny request at ingest time (not timed) so the first answer does not pay for connection setup."""
        if not self.answer_enabled or self._transport is not None or not self.ledger.allow(reserve=0.01):
            return
        try:
            import anthropic
            if self._client is None:
                self._client = anthropic.Anthropic(max_retries=0)
            extra = {"thinking": {"type": "disabled"}} if self.model.startswith("claude-haiku") else (
                {"thinking": {"type": "between_tools"}} if self.model.startswith("claude-sonnet-5-5") else {})
            resp = self._client.with_options(timeout=15, max_retries=1).messages.create(
                model=self.model, max_tokens=16 if extra else 1000, output_config={"effort": "low"},
                messages=[{"role": "user", "content": "Reply with OK."}], **extra)
            self.ledger.add("warmup", self.model, resp.usage.input_tokens or 0, resp.usage.output_tokens or 0)
        except Exception:
            pass

    def _anthropic_call(self, model: str, system: str, prompt: str, timeout: float) -> dict:
        import anthropic
        if self._client is None:
            self._client = anthropic.Anthropic(max_retries=0)
        extra = {}
        if model.startswith("claude-haiku"):
            extra["thinking"] = {"type": "disabled"}         # Haiku 5.5: allowed at effort low..high
        elif model.startswith("claude-sonnet-5-5"):
            extra["thinking"] = {"type": "between_tools"}    # Sonnet 5.5's way to turn thinking off
        # Opus 5.5: thinking cannot be disabled; low effort keeps it short
        try:
            resp = self._client.with_options(timeout=timeout, max_retries=0).messages.create(
                model=model,
                max_tokens=(110 if REASON else 24) if extra else 2000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "low"},
                **extra,
            )
        except anthropic.APITimeoutError:
            self.last_error = "timeout"
            return {"ok": False, "timeout": True}
        except anthropic.APIError as e:
            self.last_error = f"{type(e).__name__}: {getattr(e, 'message', '')}"[:300]
            return {"ok": False}
        usage = {"input_tokens": getattr(resp.usage, "input_tokens", 0) or 0,
                 "output_tokens": getattr(resp.usage, "output_tokens", 0) or 0}
        if resp.stop_reason != "end_turn":
            self.last_error = f"stop_reason {resp.stop_reason}"
            return {"ok": False, "usage": usage, "long": resp.stop_reason == "max_tokens"}
        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
        return {"ok": True, "text": text, "usage": usage}
