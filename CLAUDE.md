# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A benchmark comparing LLMs on a binary phishing-vs-legitimate email classification task. It pits Claude (Anthropic API) against local models served by Docker Model Runner (DMR), feeding every model the **exact same prompt** so the comparison is fair, then emits per-model metrics (accuracy, precision, recall, F1, avg response time, error rate) and a ranked comparison report.

Code comments, docstrings, and console output are in **Portuguese (pt-BR)**; match that when editing. The prompt itself ([prompt.py](prompt.py)) is intentionally in English.

## Commands

```bash
pip install -r requirements.txt        # install deps
python main.py                         # full benchmark: Claude + every model in LOCAL_MODELS
python tests/test_model.py             # quick 10-sample smoke test — run before a full run
python tests/test_output.py            # JSON-format consistency check for one local model (reads tests/long_email.txt)
python tests/test_parser.py            # offline unit test for the parser + heuristic JSON repair (no DMR/API)
python tests/test_resume.py            # offline test for per-email resume (fake client, temp dir)
python metrics.py --rebuild            # rebuild comparison_report.csv from existing metrics_*.json (no model re-run)
```

The scripts under `tests/` (`test_model.py`, `test_output.py`, `test_parser.py`, `test_resume.py`) are **standalone scripts, not pytest tests** — run them directly from the project root. (`test_parser.py` and `test_resume.py` are self-contained and offline; `test_model.py`/`test_output.py` need DMR.) They prepend the project root to `sys.path` so the root-level modules resolve. (The per-model run loop lives in `model_runner.py` at the project root — it is part of the main pipeline, not a test.) There is no test framework, linter, or build step configured. Python 3.10+ is required (modern type syntax like `dict | None`).

## Prerequisites to actually run a benchmark

- `ANTHROPIC_API_KEY` in `.env` (copy from `.env.example`) — only needed when `RUN_CLAUDE = True`.
- Docker Model Runner listening at `http://localhost:12434` for the local models.
- A dataset CSV with columns `id, subject, body, label` (`label`: 0 = legitimate, 1 = phishing). `config.DATASET_PATH` defaults to the bundled example file and **must be repointed** at a real dataset.

## Architecture

Single orchestrator, swappable model clients, shared prompt/parser. Flow:

`main.py` → `load_dataset()` → for each model: `run_model_tests()` → `compute_metrics()` → `save_comparison_report()`

- **[config.py](config.py)** — single source of configuration; everything imports from here. Notable flags: `RUN_CLAUDE`, `LOCAL_MODELS` (friendly name → DMR tag), `SKIP_COMPLETED_MODELS` (resume an interrupted run by skipping models that already have `metrics_<name>.json`), `ENABLE_LLM_LOGGING`, `TRUNCATE_EMAIL_BODY`/`EMAIL_BODY_MAX_CHARS`, `MAX_RETRIES`/`RETRY_DELAY`, `PHISHING_LIKELIHOOD_THRESHOLD` (the label `parse_response()` returns is derived from the numeric `phishing_likelihood >= threshold`, **not** the textual `classification` field — small models often emit a low likelihood yet a "phishing" string, and trusting the number measurably cut false positives on every local model), `USE_CONSTRAINED_DECODING` (constrains output to the shared `RESPONSE_SCHEMA` via structured-output decoding — each client wraps the same schema natively: `DmrClient` via OpenAI `response_format`, `ClaudeClient` via Anthropic `output_config.format`; the prompt still describes the format).
- **[main.py](main.py)** — runs models **sequentially** (required: DMR keeps one model in memory and swaps on request for a different tag). Calls `client.warm_up()` before each local model to avoid load-latency hitting the first email.
- **The two clients are interchangeable.** Both [models/claude_client.py](models/claude_client.py) and [models/dmr_client.py](models/dmr_client.py) expose `classify(system_prompt, user_prompt) -> dict` returning `raw_response`, `elapsed_seconds`, `input_tokens`, `output_tokens`, `error`. `DmrClient` uses the `openai` lib pointed at the local DMR endpoint (OpenAI-compatible) and adds `warm_up()`.
- **[prompt.py](prompt.py)** — `build_prompt()` returns `(system_prompt, user_prompt)` and truncates the body. The same prompt is shared across all models by design.
- **[output_parser.py](output_parser.py)** — `parse_response()` extracts the model's JSON object (tolerating markdown fences and surrounding text) via a **4-stage fallback**, then derives the internal label `PHISHING`/`LEGÍTIMO` from the numeric `phishing_likelihood` via `_decide_label()` (`>= PHISHING_LIKELIHOOD_THRESHOLD` → PHISHING). The textual `classification` field is read only as a **fallback** when no usable likelihood is present, and to **note divergences** in `parse_note` (it stays in the schema/prompt because committing to a label helps the model reason). Stages 1–2 (whole-text / first `{...}`) just **locate** already-valid JSON. Stages 3–4 **repair** malformed JSON: `_close_json()` closes JSON truncated by the token limit, and `_escape_inner_quotes()` escapes unescaped literal double-quotes inside strings (a common small-model bug). `_extract_json()` returns `(data, repaired)`; only stages 3–4 set `repaired=True`, surfaced as `parse_response()["repaired"]`.
- **[model_runner.py](model_runner.py)** — `run_model_tests(..., resume=False)` iterates the dataset, calls the client, parses, and logs each row. With `resume=True` it calls `load_partial_results()` first to skip already-logged emails.
- **[logger.py](logger.py)** — `BenchmarkLogger` writes one CSV per model run (timestamped) when logging is enabled, **appending each row immediately** (not at the end). `load_partial_results()` enables per-email resume: it reads the model's latest partial log, drops/re-tests the last row, rewrites the CSV without it, and returns the reusable records + the email ids to skip (so the resumed run continues in the *same* log file).
- **[metrics.py](metrics.py)** — sklearn metrics with **PHISHING (1) as the positive class**; ranks the comparison report by F1. Emits **two metric tracks** per model: the **strict/raw** track (existing keys: `error_rate`, `accuracy`, `f1_score`, ... — heuristic-repaired responses count as errors and are excluded from accuracy) and the **rescued** track (`rescued_*` keys — repaired predictions are folded in; only `predicted_label == -1` remains an error). `repaired_count` reports how many were rescued. `_classification_metrics()` is the shared helper computing accuracy/precision/recall/F1/confusion for a record subset.

## Error / label conventions (easy to get wrong)

- Three outcomes in `BenchmarkLogger.log` ([logger.py](logger.py)), where the `is_error`/`predicted_label`/`was_repaired` logic lives (not the parser):
  - **Clean** — parsed without repair: `is_error = False`, `was_repaired = False`, `predicted_label ∈ {0,1}`.
  - **Repaired** — only readable after heuristic repair (`parse_response()["repaired"]`): `is_error = True` **and** `was_repaired = True`, but `predicted_label` keeps the rescued `{0,1}` value (**not** `-1`). The raw output wasn't valid JSON, so it counts as an error, yet the rescued prediction is preserved for the rescued track.
  - **Failed/unparseable** — client `error` or `predicted is None` even after repair: `predicted_label = -1`, `is_error = True`, `was_repaired = False`.
- Metric consequence: the **strict** track uses only clean records (`not is_error`) → repaired counts in `error_rate`, excluded from accuracy. The **rescued** track uses any usable prediction (`predicted_label != -1`) → repaired folded into `rescued_accuracy`, and only the failed/unparseable remain in `rescued_error_rate`. `avg_time_sec` always covers all rows.
- Clients **never raise per-email**; they return `error` in the dict. A failing **local** model does not stop the benchmark — `main.py` moves on. A failing **Claude** run calls `sys.exit(1)`.
- Two independent resume mechanisms (config flags): `SKIP_COMPLETED_MODELS` (default `False`) skips whole models that already have `metrics_<model>.json`; `RESUME_PARTIAL_MODEL` (default `True`) resumes the *partially-done* model from its partial log (re-testing only its last email). They compose. To force a clean re-run of a model, delete its CSV in `results/llm_logs/`.
- All outputs go to `results/` (gitignored): `results/reports/metrics_<model>.json`, `results/reports/comparison_report.csv`, `results/llm_logs/<model>_<timestamp>.csv`.

## Notes

- `legacy/` holds old JS/Python scripts outside the current pipeline — ignore unless explicitly asked.
