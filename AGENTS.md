# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

LLM Parse Service — a FastAPI microservice that accepts a prompt + JSON Schema and returns structured data from an OpenAI-compatible LLM. Single Python package, no database, no frontend.

### Running commands

- **Package manager:** `uv` (lockfile: `uv.lock`)
- **Lint:** `uv run ruff check .` and `uv run ruff format --check .`
- **Type check:** `uv run mypy src/`
- **Tests:** `uv run pytest tests/ -v` (all tests are fully mocked, no LLM calls)
- **Dev server:** `uv run uvicorn llm_parse.api.app:app --reload --host 0.0.0.0 --port 8000`

See `README.md` for full API usage and configuration reference.

### Gotchas

- **Tests vs injected secrets:** The unit test `test_default_values` in `tests/unit/test_config.py` asserts the default `OPENAI_BASE_URL`. If `OPENAI_BASE_URL` is set as an environment variable (e.g. via injected secrets), this test will fail. Run tests with `env -u OPENAI_BASE_URL -u MODEL_NAME` to avoid env var interference, or set `OPENAI_API_KEY=test-key-for-ci` and unset the others.
- **`.env` file:** The service reads `.env` via `pydantic-settings`. Copy `.env.example` to `.env` for local config. For tests, the `.env` file is generally not needed since tests mock everything.
- **Python version:** Requires Python >= 3.11. The system Python 3.12 works fine.
