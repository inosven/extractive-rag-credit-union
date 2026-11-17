"""Shared helpers for the LLM-written evaluation set.

Independence rule: nothing in tools/evalgen imports submission/ (the system under test) except the spend ledger
class in submission/llm.py, which holds no answering logic. Pages are read straight from snapshots/ with the
harness loader; scoring.contains is the harness's own matcher.

All outputs quote website text, so they go to PRIVATE (outside the git tree), never into the repository.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))

PRIVATE = Path(os.getenv("DUPACO_EVAL_PRIVATE", str(ROOT.parent / "llm_private" / "evalset")))
KEY_FILE = Path(os.getenv("DUPACO_LLM_KEY_FILE", str(ROOT.parent / ".anthropic_key")))
LEDGER = Path(os.getenv("DUPACO_LLM_CACHE_DIR", str(ROOT / ".llm_cache"))) / "spend.jsonl"
CAP_USD = float(os.getenv("DUPACO_LLM_BUDGET_USD", "95"))

GEN_MODEL = "claude-opus-5-5"
CHECK_MODEL = "claude-sonnet-5-5"


def ledger():
    from llm import SpendLedger
    return SpendLedger(LEDGER, CAP_USD)


def client():
    """Anthropic client with the key from the environment or the key file. Never prints the key."""
    import anthropic
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        from llm import _read_key_file
        key = _read_key_file(str(KEY_FILE))
        if not key:
            raise SystemExit(f"no API key: create {KEY_FILE} with a line ANTHROPIC_API_KEY=...")
        os.environ["ANTHROPIC_API_KEY"] = key
    return anthropic.Anthropic()


# ---- snapshots, read independently of the submission's parser

def snapshots():
    """[(snapshot_id, taken_on, {url: (title, markdown)})], oldest first."""
    out = []
    for d in sorted(p for p in (ROOT / "snapshots").iterdir() if (p / "manifest.json").exists()):
        man = json.loads((d / "manifest.json").read_text())
        pages = {}
        for p in man["pages"]:
            pages[p["url"]] = (p.get("title", ""), (d / p["file"]).read_text(encoding="utf-8"))
        out.append((man["snapshot_id"], man["taken_on"], pages))
    out.sort(key=lambda x: x[1])
    return out


def snapshot_for(snaps, as_of: str | None):
    """Newest snapshot taken on or before as_of (the first one if as_of is earlier); latest when as_of is None."""
    if not as_of:
        return snaps[-1]
    ok = [s for s in snaps if s[1] <= as_of]
    return ok[-1] if ok else snaps[0]


def clean(md: str) -> str:
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)                 # images
    md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)              # links -> anchor text
    md = re.sub(r"&amp;#45;|&#45;", "-", md).replace("&amp;", "&")
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def kind(url: str) -> str:
    u = url.lower()
    if "fee-schedule" in u or "/fees" in u:
        return "fees"
    if re.search(r"/rates\b|business-rates", u):
        return "rates"
    if "/newsroom/" in u or "/news/" in u or re.search(r"/20\d\d/\d\d/", u):
        return "news"
    if "/learn/" in u or "/blog/" in u:
        return "learn"
    if "calculator" in u:
        return "calculator"
    if re.search(r"/personal/|/business/", u):
        return "product"
    if re.search(r"contact|locat|branch|hours", u):
        return "branch"
    return "other"


def jdump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def jload(path: Path):
    return json.loads(path.read_text())


# ---- batch helpers

def estimate_tokens(text: str) -> int:
    return int(len(text) / 3.5) + 1


def guard_budget(est_usd: float) -> None:
    led = ledger()
    if not led.allow(reserve=est_usd * 1.5):
        raise SystemExit(f"budget guard: recorded ${led.total:.2f} + estimate ${est_usd:.2f} x1.5 would pass the "
                         f"${CAP_USD:.0f} cap")
    print(f"budget: recorded ${led.total:.2f}, this step estimated ${est_usd:.2f}, cap ${CAP_USD:.0f}")


def run_batch(requests: list, purpose: str, state_file: Path) -> dict:
    """Submit (or resume) a batch, wait for it, record spend at the 50% batch rate. Returns {custom_id: text|None}."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    c = client()
    state = jload(state_file) if state_file.exists() else {}
    if not state.get("batch_id"):
        b = c.messages.batches.create(requests=[Request(custom_id=r["custom_id"],
                                                        params=MessageCreateParamsNonStreaming(**r["params"]))
                                                for r in requests])
        state = {"batch_id": b.id, "purpose": purpose, "n": len(requests), "recorded": False}
        jdump(state, state_file)
        print(f"submitted batch {b.id} with {len(requests)} requests")
    while True:
        b = c.messages.batches.retrieve(state["batch_id"])
        if b.processing_status == "ended":
            break
        print(f"  {b.processing_status}: {b.request_counts.processing} processing, "
              f"{b.request_counts.succeeded} done", flush=True)
        time.sleep(60)
    out, led = {}, ledger()
    tin = tout = 0
    model = requests[0]["params"]["model"] if requests else GEN_MODEL
    for r in c.messages.batches.results(state["batch_id"]):
        if r.result.type == "succeeded":
            m = r.result.message
            tin += m.usage.input_tokens or 0
            tout += m.usage.output_tokens or 0
            text = next((b.text for b in m.content if getattr(b, "type", "") == "text"), "")
            out[r.custom_id] = text if m.stop_reason == "end_turn" else None
        else:
            out[r.custom_id] = None
    if not state.get("recorded"):
        from llm import cost_usd
        usd = cost_usd(model, tin, tout) * 0.5
        led.add(purpose, model, tin, tout, usd=usd, note=f"batch {state['batch_id']}")
        state["recorded"] = True
        state["usd"] = round(usd, 4)
        jdump(state, state_file)
        print(f"batch spend ${usd:.3f} ({tin} in, {tout} out); ledger total ${led.total:.2f}")
    return out
