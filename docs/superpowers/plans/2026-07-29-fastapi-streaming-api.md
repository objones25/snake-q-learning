# FastAPI Streaming API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose `train()`/`play()` over a streaming HTTP API (SSE) so a Railway-deployed frontend can render live Q-learning training or greedy gameplay as a snake game in the browser.

**Architecture:** A new `api.py` module (same layer as `main.py`/`watch.py`) wraps the existing `train()`/`play()` generators in async SSE-streaming FastAPI endpoints. `train()` loses its file-save side effect — saving moves to `main.py`'s CLI caller — so the API can stream training runs without ever writing to disk. `/play` always loads a new committed `example_q_table.json` (since the real `q_table.json` is gitignored and won't exist in a fresh deployment).

**Tech Stack:** FastAPI, `uvicorn[standard]`, `httpx` (for `TestClient`) — all behind a new `api` extra in `pyproject.toml`, following the same optional-dependency pattern as `render`/`plot`.

## Global Constraints

- Python `>=3.13`, package manager is `uv` (per `CLAUDE.md`).
- No behavior change to `main.py`'s CLI output/flags — `main.py train` must still write `q_table.json` exactly as it does today.
- `/play`'s Q-table path is never client-controlled (avoids path traversal on a public endpoint).
- `/train` and `/play` cap `n_episodes` (≤200 / ≤100) and `grid_size` (8–40) via FastAPI `Query(..., ge=, le=)`, returning normal 422s when exceeded.
- CORS wide open (`allow_origins=["*"]`) — no auth, no per-origin config.
- Existing test conventions: plain `pytest`, `Test*` classes grouping by behavior, no `conftest.py`/fixtures, helper functions instead.

---

### Task 1: Make `train()` side-effect-free; move saving into `main.py`

**Files:**

- Modify: `train.py:32` (remove the `agent.save(...)` line)
- Modify: `main.py:47-67` (`_run_train`)
- Modify: `tests/test_train.py:33-37` (remove `test_saves_q_table_to_path`)
- Modify: `tests/test_main.py` (add a CLI-level save assertion)

**Interfaces:**

- Consumes: existing `train(config: TrainConfig) -> Iterator[EpisodeStep]`, `EpisodeStep.agent: QLearningAgent`, `QLearningAgent.save(path: Path) -> None`.
- Produces: `train()` with identical signature/yields but no filesystem writes — `api.py` (Task 4) calls it directly with no save step.

- [ ] **Step 1: Write the failing test for CLI-level saving**

Add to `tests/test_main.py`, in `class TestTrainAndPlayPrintProgress` (or a new class right after it):

```python
class TestTrainCommandSavesQTable:
    def test_train_command_saves_q_table_to_path(self, tmp_path):
        path = tmp_path / "q_table.json"
        main.main(["train", "--n-episodes", "20", "--grid-size", "8", "--save-path", str(path)])
        assert path.exists()
```

- [ ] **Step 2: Run it to confirm it currently passes (baseline before refactor)**

Run: `uv run pytest tests/test_main.py::TestTrainCommandSavesQTable -v`
Expected: PASS (today's `train()` still saves, so this passes before you touch anything — it's the regression guard for the refactor you're about to do).

- [ ] **Step 3: Remove the save call from `train()`**

In `train.py`, delete the last line of the function body:

```python
    agent.save(config.save_path)
```

so `train()` ends right after the `for episode in range(config.n_episodes):` loop, with no trailing statement.

- [ ] **Step 4: Run `tests/test_train.py` and confirm `test_saves_q_table_to_path` now fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: `TestTrain::test_saves_q_table_to_path` FAILS (`assert path.exists()` is False); all other tests in the file still pass.

- [ ] **Step 5: Delete the now-invalid test from `tests/test_train.py`**

Remove this whole method from `class TestTrain`:

```python
    def test_saves_q_table_to_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q_table.json"
            list(train(TrainConfig(n_episodes=20, grid_size=8, save_path=path)))
            assert path.exists()
```

- [ ] **Step 6: Update `main.py`'s `_run_train` to save after the loop**

In `main.py`, replace:

```python
def _run_train(config: TrainConfig, plot: bool = False, plot_path: Path | None = None) -> None:
    recent_scores: deque[int] = deque(maxlen=500)
    top_score = 0
    checkpoints: list[tuple[int, float]] = []
    for step in train(config):
        if not step.result.done:
            continue
        recent_scores.append(step.result.info["score"])
        top_score = max(top_score, step.result.info["score"])
        if step.episode % 500 == 0:
            avg_score = sum(recent_scores) / len(recent_scores)
            checkpoints.append((step.episode, avg_score))
            print(
                f"episode {step.episode:6d}  epsilon={step.agent.epsilon:.3f}  "
                f"avg_score={avg_score:.2f}  top_score={top_score}"
            )

    if plot and checkpoints:
        from plotting import plot_training_progress

        plot_training_progress(checkpoints, save_path=plot_path)
```

with:

```python
def _run_train(config: TrainConfig, plot: bool = False, plot_path: Path | None = None) -> None:
    recent_scores: deque[int] = deque(maxlen=500)
    top_score = 0
    checkpoints: list[tuple[int, float]] = []
    agent = None
    for step in train(config):
        agent = step.agent
        if not step.result.done:
            continue
        recent_scores.append(step.result.info["score"])
        top_score = max(top_score, step.result.info["score"])
        if step.episode % 500 == 0:
            avg_score = sum(recent_scores) / len(recent_scores)
            checkpoints.append((step.episode, avg_score))
            print(
                f"episode {step.episode:6d}  epsilon={step.agent.epsilon:.3f}  "
                f"avg_score={avg_score:.2f}  top_score={top_score}"
            )

    if agent is not None:
        agent.save(config.save_path)

    if plot and checkpoints:
        from plotting import plot_training_progress

        plot_training_progress(checkpoints, save_path=plot_path)
```

- [ ] **Step 7: Run the new CLI-level test and confirm it passes again**

Run: `uv run pytest tests/test_main.py::TestTrainCommandSavesQTable -v`
Expected: PASS

- [ ] **Step 8: Run the full suite to confirm nothing else broke**

Run: `uv run pytest -v`
Expected: All tests PASS (the deleted `test_saves_q_table_to_path` is gone, everything else green).

- [ ] **Step 9: Commit**

```bash
git add train.py main.py tests/test_train.py tests/test_main.py
git commit -m "Move q_table save out of train() into main.py's CLI caller

train() is now fully side-effect-free like play(), so it can be streamed
by an HTTP API without ever writing to disk. main.py's _run_train saves
the final agent after the loop instead, preserving today's CLI behavior."
```

---

### Task 2: Commit `example_q_table.json`

**Files:**

- Create: `example_q_table.json` (copy of the current `q_table.json`)

**Interfaces:**

- Produces: a file at repo root, valid JSON, `list[list[float]]` shape with exactly `SnakeState.N_STATES` (1600) rows of 3 floats each — loadable via `QLearningAgent.load`. Task 4's `/play` endpoint hardcodes this path.

- [ ] **Step 1: Copy the current trained table**

```bash
cp q_table.json example_q_table.json
```

- [ ] **Step 2: Confirm it's not excluded by `.gitignore`**

Run: `git check-ignore -v example_q_table.json`
Expected: no output (exit code 1) — the `.gitignore` rule is an exact match on the literal name `q_table.json`, so `example_q_table.json` isn't caught by it. If this prints a matching rule, stop and re-check `.gitignore` before proceeding — do not force-add with `-f` without understanding why it matched.

- [ ] **Step 3: Sanity-check the shape matches `SnakeState.N_STATES`**

```bash
uv run python -c "
import json
from snake_state import SnakeState
data = json.load(open('example_q_table.json'))
assert len(data) == SnakeState.N_STATES, (len(data), SnakeState.N_STATES)
assert all(len(row) == 3 for row in data)
print('OK', len(data), 'rows')
"
```

Expected: prints `OK 1600 rows`.

- [ ] **Step 4: Commit**

```bash
git add example_q_table.json
git commit -m "Commit example_q_table.json for the deployed /play API endpoint

q_table.json itself is gitignored (it's a local training artifact), so a
fresh deployment has no table to load. This is a one-time snapshot of a
trained table, not kept in sync with future retraining."
```

---

### Task 3: Add the `api` optional-dependency group

**Files:**

- Modify: `pyproject.toml`

**Interfaces:**

- Produces: `uv sync --extra api` installs `fastapi`, `uvicorn[standard]`, `httpx` — required by Task 4 (`api.py`) and Task 5 (`tests/test_api.py`, which needs `httpx` for `TestClient`).

- [ ] **Step 1: Edit `pyproject.toml`**

Change:

```toml
[project.optional-dependencies]
render = ["pygame>=2.5"]
plot = ["matplotlib>=3.11.1"]
```

to:

```toml
[project.optional-dependencies]
render = ["pygame>=2.5"]
plot = ["matplotlib>=3.11.1"]
api = ["fastapi>=0.115", "uvicorn[standard]>=0.32", "httpx>=0.27"]
```

- [ ] **Step 2: Sync and verify install**

Run: `uv sync --extra api`
Expected: exits 0, `uv.lock` updates to include `fastapi`, `uvicorn`, `httpx` and their transitive deps.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add optional api dependency group (fastapi, uvicorn, httpx)"
```

---

### Task 4: `api.py` — the FastAPI app with `/train` and `/play` SSE endpoints

**Files:**

- Create: `api.py`

**Interfaces:**

- Consumes: `train(config: TrainConfig) -> Iterator[EpisodeStep]`, `play(config: PlayConfig) -> Iterator[EpisodeStep]`, `EpisodeStep.episode: int`, `EpisodeStep.result: StepResult`, `EpisodeStep.agent.epsilon: float`, `StepResult.board.grid_size/snake_body/food`, `StepResult.info["score"]`, `StepResult.reward`, `StepResult.done`, `TrainConfig(n_episodes, grid_size, agent: AgentConfig)`, `AgentConfig(alpha, gamma, epsilon_start, epsilon_end, epsilon_decay_episodes)`, `PlayConfig(n_episodes, grid_size, q_table_path)`.
- Produces: module-level `app: FastAPI` object with `GET /train` and `GET /play`. This is what `uvicorn api:app` (Task 6's `Procfile`) and `tests/test_api.py` (Task 5) import.

- [ ] **Step 1: Write `api.py`**

```python
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import AgentConfig, PlayConfig, TrainConfig
from episode_step import EpisodeStep
from play import play
from train import train

EXAMPLE_Q_TABLE_PATH = Path("example_q_table.json")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])


def _encode(step: EpisodeStep, epsilon: float | None) -> str:
    board = step.result.board
    payload = {
        "episode": step.episode,
        "board": {
            "grid_size": board.grid_size,
            "snake_body": board.snake_body,
            "food": board.food,
        },
        "score": step.result.info["score"],
        "reward": step.result.reward,
        "done": step.result.done,
        "epsilon": epsilon,
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_train(config: TrainConfig, render_every: int, fps: float):
    delay = 1 / fps
    for step in train(config):
        if step.episode % render_every != 0:
            continue
        yield _encode(step, epsilon=step.agent.epsilon)
        await asyncio.sleep(delay)


async def _stream_play(config: PlayConfig, fps: float):
    delay = 1 / fps
    for step in play(config):
        yield _encode(step, epsilon=None)
        await asyncio.sleep(delay)


@app.get("/train")
def stream_train(
    n_episodes: int = Query(50, ge=1, le=200),
    grid_size: int = Query(20, ge=8, le=40),
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay_episodes: int = 5_000,
    render_every: int = Query(1, ge=1),
    fps: float = Query(30.0, gt=0),
) -> StreamingResponse:
    config = TrainConfig(
        n_episodes=n_episodes,
        grid_size=grid_size,
        agent=AgentConfig(
            alpha=alpha,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_episodes=epsilon_decay_episodes,
        ),
    )
    return StreamingResponse(
        _stream_train(config, render_every, fps), media_type="text/event-stream"
    )


@app.get("/play")
def stream_play(
    n_episodes: int = Query(10, ge=1, le=100),
    grid_size: int = Query(20, ge=8, le=40),
    fps: float = Query(10.0, gt=0),
) -> StreamingResponse:
    config = PlayConfig(
        n_episodes=n_episodes, grid_size=grid_size, q_table_path=EXAMPLE_Q_TABLE_PATH
    )
    return StreamingResponse(_stream_play(config, fps), media_type="text/event-stream")
```

- [ ] **Step 2: Verify it imports and serves without error**

Run:

```bash
uv run --extra api python -c "
from api import app
print(sorted(r.path for r in app.routes if r.path in ('/train', '/play')))
"
```

Expected: `['/play', '/train']`

- [ ] **Step 3: Commit**

```bash
git add api.py
git commit -m "Add FastAPI app with SSE-streaming /train and /play endpoints"
```

---

### Task 5: Tests for `api.py`

**Files:**

- Create: `tests/test_api.py`

**Interfaces:**

- Consumes: `api.app` (Task 4), `fastapi.testclient.TestClient`.
- Produces: nothing consumed by later tasks — this is a leaf test module.

- [ ] **Step 1: Write `tests/test_api.py`**

```python
import json

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _parse_sse_frames(body: str) -> list[dict]:
    frames = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        assert chunk.startswith("data: ")
        frames.append(json.loads(chunk[len("data: ") :]))
    return frames


class TestTrainEndpoint:
    def test_streams_sse_frames_with_expected_keys(self):
        response = client.get(
            "/train", params={"n_episodes": 1, "grid_size": 8, "fps": 1000}
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        frames = _parse_sse_frames(response.text)
        assert len(frames) > 0
        for frame in frames:
            assert frame.keys() == {"episode", "board", "score", "reward", "done", "epsilon"}
            assert frame["board"].keys() == {"grid_size", "snake_body", "food"}
            assert frame["epsilon"] is not None

    def test_n_episodes_over_cap_is_rejected(self):
        response = client.get("/train", params={"n_episodes": 201})
        assert response.status_code == 422

    def test_grid_size_out_of_range_is_rejected(self):
        response = client.get("/train", params={"grid_size": 4})
        assert response.status_code == 422


class TestPlayEndpoint:
    def test_streams_sse_frames_with_null_epsilon(self):
        response = client.get(
            "/play", params={"n_episodes": 1, "grid_size": 8, "fps": 1000}
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        frames = _parse_sse_frames(response.text)
        assert len(frames) > 0
        for frame in frames:
            assert frame.keys() == {"episode", "board", "score", "reward", "done", "epsilon"}
            assert frame["epsilon"] is None

    def test_n_episodes_over_cap_is_rejected(self):
        response = client.get("/play", params={"n_episodes": 101})
        assert response.status_code == 422

    def test_grid_size_out_of_range_is_rejected(self):
        response = client.get("/play", params={"grid_size": 41})
        assert response.status_code == 422
```

- [ ] **Step 2: Run the tests**

Run: `uv run --extra api pytest tests/test_api.py -v`
Expected: all PASS. (`TestClient` drives the app synchronously — `StreamingResponse`'s async generator is fully consumed before `response.text` is available, so no manual event-loop handling is needed. `fps=1000` keeps the per-step `asyncio.sleep` calls at 1ms, so `TestPlayEndpoint`/`TestTrainEndpoint`'s streaming tests stay fast even though `/play`'s default `fps=10` and `/train`'s default `fps=30` would otherwise make a test with several steps take real wall-clock seconds.)

If `test_streams_sse_frames_with_expected_keys` fails on the `/play` test with a `FileNotFoundError`, confirm Task 2's `example_q_table.json` exists at the repo root and Task 4's `EXAMPLE_Q_TABLE_PATH = Path("example_q_table.json")` matches its actual filename exactly.

- [ ] **Step 3: Run the full suite one more time**

Run: `uv run --extra api pytest -v`
Expected: all PASS (this now includes `test_api.py` alongside every other test file).

- [ ] **Step 4: Commit**

```bash
git add tests/test_api.py
git commit -m "Add tests for /train and /play SSE endpoints"
```

---

### Task 6: `Procfile` and documentation

**Files:**

- Create: `Procfile`
- Modify: `README.md` (Quick start section)
- Modify: `CLAUDE.md` ("What this is" / architecture description)

**Interfaces:**

- Consumes: Task 3's `api` extra, Task 4's `api:app`.
- Produces: nothing consumed by later tasks — this is documentation/deployment config, the final task in the plan.

- [ ] **Step 1: Write `Procfile`**

```
web: uv run --extra api uvicorn api:app --host 0.0.0.0 --port $PORT
```

- [ ] **Step 2: Verify it runs locally**

Run (in the background, then curl it, then stop it):

```bash
PORT=8000 uv run --extra api uvicorn api:app --host 0.0.0.0 --port $PORT &
SERVER_PID=$!
sleep 1
curl -sN "http://localhost:8000/play?n_episodes=1&grid_size=8&fps=1000" | head -c 300
echo
kill $SERVER_PID
```

Expected: the `curl` output starts with `data: {"episode": 0, "board": {...`.

- [ ] **Step 3: Add an API section to `README.md`**

In `README.md`, after the existing `uv sync --extra plot ...` block in the Quick start section, add:

```markdown
uv sync --extra api # install optional fastapi/uvicorn dependencies
uv run --extra api uvicorn api:app --reload # serve /train and /play as SSE streams on :8000
```

And after the paragraph describing `--plot`, add:

```markdown
`api.py` is a separate, optional HTTP entry point (`uv sync --extra api`) that
exposes `train()`/`play()` as Server-Sent Events over `GET /train` and
`GET /play` — for driving a browser-rendered snake game (e.g. deployed to
Railway via the included `Procfile`). `/train` streams live training,
`/play` always plays back the committed `example_q_table.json` (the real
`q_table.json` is gitignored, so a fresh deployment has nothing else to
load). Neither endpoint writes to disk.
```

- [ ] **Step 4: Update `CLAUDE.md`**

In the "What this is" paragraph, after the sentence about `--plot`/`--plot-path` and `plotting.py`, add:

```markdown
A third optional entry point, `api.py` (FastAPI, `uv sync --extra api`), exposes `train()`/`play()` over HTTP as Server-Sent Events (`GET /train`, `GET /play`) for a browser frontend — deployable to Railway via the included `Procfile`. `train()` itself no longer writes `q_table.json`; that save now happens in `main.py`'s `_run_train` after the generator is exhausted, so `api.py` can stream a training run without touching disk. `/play` always loads the committed `example_q_table.json` rather than the gitignored `q_table.json`, since a fresh deployment has no trained table of its own.
```

Add a corresponding line to the Commands section's code block:

```bash
uv sync --extra api                        # install optional fastapi/uvicorn dependencies
uv run --extra api uvicorn api:app --reload  # serve /train and /play as SSE streams on :8000
uv run --extra api pytest tests/test_api.py  # api-specific tests (needs httpx from the api extra)
```

- [ ] **Step 5: Commit**

```bash
git add Procfile README.md CLAUDE.md
git commit -m "Add Procfile and document the api.py FastAPI entry point"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers the spec's `train()` side-effect removal; Task 2 covers `example_q_table.json`; Tasks 3–4 cover the new module, deps, and endpoint contracts (SSE payload shape, caps, CORS, no client-controlled `q_table_path`); Task 5 covers the spec's testing-impact section; Task 6 covers `Procfile`/deployment and docs. No spec section is unaddressed.
- **Placeholder scan:** no TBD/TODO; every step has literal code or an exact command.
- **Type consistency:** `TrainConfig`/`PlayConfig`/`AgentConfig` field names in Task 4 match `config.py` exactly (verified against the file read during brainstorming); `EpisodeStep.result.board.{grid_size,snake_body,food}` and `StepResult.{reward,done,info}` match `snake_env.py`/`episode_step.py`; `QLearningAgent.epsilon` matches `q_agent.py`. Test helper `_parse_sse_frames` in Task 5 matches the exact `data: {...}\n\n` framing `api.py`'s `_encode` produces in Task 4.
