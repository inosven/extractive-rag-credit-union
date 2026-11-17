# tools/: development scripts for the optional LLM layer

Nothing here is needed to run or grade the submission. The default system (`DUPACO_LLM` unset) is the
deterministic, standard-library-only one, and `tests/test_llm_off.py` checks that it answers exactly like `main`.

## Setup

    python3 -m venv ../.venv && ../.venv/bin/pip install -r submission/requirements-llm.txt

API key: put one line `ANTHROPIC_API_KEY=sk-ant-...` in the file `../.anthropic_key` (the folder that holds this
repository, outside the git tree), `chmod 600` it. `submission/llm.py` reads it through `DUPACO_LLM_KEY_FILE`;
the tools default that variable to this path. The key is never printed or logged. `ANTHROPIC_API_KEY` in the
environment works too and takes precedence.

## Switching the layer on

    DUPACO_LLM=answer DUPACO_LLM_KEY_FILE=../.anthropic_key DUPACO_LLM_CACHE_DIR=.llm_cache \
        python harness/run_eval.py --submission submission --eval dev/open_eval.jsonl --out results_llm

`.llm_cache/` (git-ignored) holds the selection cache and `spend.jsonl`, the spend ledger. Every call is recorded
there and no call is made once `DUPACO_LLM_BUDGET_USD` (default 95) is reached. The cache quotes website text:
never commit it or copy it into a public place.

## Scripts

| Script | API? | What it does |
|---|---|---|
| `dump_answers.py` | only if `DUPACO_LLM` is set | every answer and `conflicts()` of a submission folder as JSON |
| `path_stats.py` | only if `DUPACO_LLM` is set | how many questions take each answer path, and the score per path |
| `probe_latency.py` | yes (`--dry-run`: no) | latency p50/p95 and cost of the selector per model on real fallback questions |
| `recall_stats.py` | only if `DUPACO_LLM` is set | where answers are lost: candidate recall@3/6/12, then per question "not in candidates" or "there but not chosen"; per rule handler how many answers the review fixed or broke |
| `expand_passages.py` | yes, Batch, Claude Haiku 5.5 (`--dry-run`: no) | document expansion: three questions per passage, written once into `.llm_cache/doc2query.jsonl`; used when `DUPACO_EXPAND=on` |
| `milestone.py` | only with `--llm answer` | scores several question sets in one mode; `--set-env DUPACO_PASSAGES=off` and similar switches for ablations |

## The LLM-written eval set (`tools/evalgen/`)

Independent of the system under test: these scripts read only `snapshots/` and `harness/` (plus the spend
ledger class in `submission/llm.py`). Every output goes to `../llm_private/evalset/` (outside the repository,
override with `DUPACO_EVAL_PRIVATE`) because it quotes website text.

| Step | API? | |
|---|---|---|
| `prepare.py` | no | picks ~110 pages of the newest snapshot (stratified by page kind) and ~40 table rows whose value changed between snapshots (own plain table diff) -> `tasks.json` |
| `gen.py` | yes, Batch, Claude Opus 5.5 | 2-3 items per task (plus paraphrases); `--dry-run` estimates cost -> `raw_items.jsonl` |
| `verify.py` | machine checks no; model check yes, Batch, Claude Sonnet 5.5 | tokens checked with `scoring.contains` on the source page in the snapshot that applies to `as_of`; traps must occur in the data; should-refuse terms must be absent; then an independent model audit -> `verified.jsonl` |
| `finalize.py` | no | paraphrase groups, near-duplicate removal against the public and dev sets, group-level split into `llm_dev.jsonl` and sealed `llm_test.jsonl`, sha256 in `SPLIT.md` |
