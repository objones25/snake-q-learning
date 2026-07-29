# FastAPI Streaming API Design

## Problem

`train()`/`play()` are already headless generators, but the only consumers are
CLI entry points (`main.py`, `watch.py`) that either print to a terminal or
render to a local pygame window. There's no way to drive a snake game on a
website: nothing serves the episode stream over HTTP, and `train()` currently
bakes in a file-write side effect (`agent.save(config.save_path)` at the end
of the loop) that a public, stateless API request has no business performing.

## Goals

- A small HTTP API, deployable to Railway, that streams live `train()`/`play()`
  episode steps to a browser so a frontend can render them as a snake game.
- `train()` becomes fully side-effect-free (no file write), matching `play()`.
  CLI behavior (`main.py train` still writing `q_table.json`) is unchanged —
  the save moves to the caller, not away entirely.
- `play()`'s API endpoint uses a Q-table that's actually available in a fresh
  deployment (checked into git), since `q_table.json` is gitignored.
- Safe to expose publicly: no client-controlled filesystem paths, no
  unbounded-duration requests.

## Non-goals

- No persistence, database, or auth. No way for a client to save a table it
  trained via the API — the streamed run simply ends.
- No WebSocket / bidirectional control (pause, rewind) — pure one-way
  playback, matching how `watch.py` already works.
- No change to `SnakeEnv`, `QLearningAgent`, `SnakeState`, or the state
  encoding — this is purely a new consumer of the existing `train()`/`play()`
  generators plus one refactor to `train()`'s side effect.

## Design

### `train()` becomes side-effect-free

`train.py` drops the `agent.save(config.save_path)` call at the end of the
loop. `train(config: TrainConfig) -> Iterator[EpisodeStep]` no longer touches
the filesystem at all, same as `play()` already doesn't.

`main.py`'s `_run_train` becomes responsible for saving, since it's the only
caller that needs the table persisted:

```python
def _run_train(config: TrainConfig, plot: bool = False, plot_path: Path | None = None) -> None:
    recent_scores: deque[int] = deque(maxlen=500)
    top_score = 0
    checkpoints: list[tuple[int, float]] = []
    agent = None
    for step in train(config):
        agent = step.agent
        ...  # unchanged printing/checkpoint logic
    if agent is not None:
        agent.save(config.save_path)
    ...  # unchanged plotting
```

`step.agent` is the same mutable `QLearningAgent` instance on every yield, so
capturing it once (or just overwriting `agent` each iteration) and saving
after the loop reproduces today's CLI behavior exactly. `TrainConfig.save_path`
is untouched — it's still a real field, just read by `main.py` now instead of
`train.py`.

`api.py`'s `/train` endpoint calls `train(config)` directly and never saves.

### `example_q_table.json`

Copy the current `q_table.json` to `example_q_table.json` and commit it. The
`.gitignore` rule is an exact match on `q_table.json`, so the new file isn't
excluded. This is a one-time snapshot of a trained table for the deployed
demo — not kept in sync with future retraining unless someone deliberately
re-copies it.

### New module: `api.py`

Sits at the same layer as `main.py`/`watch.py` — imports `train()`, `play()`,
and `config.py` directly; nothing else imports `api.py`.

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
):
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
):
    config = PlayConfig(
        n_episodes=n_episodes, grid_size=grid_size, q_table_path=EXAMPLE_Q_TABLE_PATH
    )
    return StreamingResponse(_stream_play(config, fps), media_type="text/event-stream")
```

`q_table_path` is deliberately **not** a request parameter on `/play` —
accepting a client-supplied filesystem path on a public endpoint is a
path-traversal / arbitrary-file-read risk, and there's no notion of "the
client's own table" in this deployment anyway. The endpoint always loads
`EXAMPLE_Q_TABLE_PATH`.

`n_episodes` and `grid_size` are capped via FastAPI's `Query(..., ge=, le=)`
on both endpoints — an out-of-range value is a normal 422, not a silently
clamped value. `render_every`/`fps` are API-only pacing knobs with no CLI
equivalent; `alpha`/`gamma`/`epsilon_*` on `/train` are uncapped since they
don't affect stream duration or size, and mirror the CLI's existing
`--alpha`/`--gamma`/etc. flags one-for-one.

### SSE payload shape

One SSE frame per yielded `EpisodeStep` (after `render_every` filtering on
`/train`), default `message` event, no custom event name:

```
data: {"episode": 3, "board": {"grid_size": 20, "snake_body": [[10,10],[10,11]], "food": [4,7]}, "score": 4, "reward": 0, "done": false, "epsilon": 0.87}

```

`epsilon` is populated on `/train` (reflects live decay) and `null` on
`/play` (always greedy, `epsilon=0`, not informative). `snake_body`/`food`
tuples serialize as JSON arrays via `json.dumps` — no custom encoder needed
since `Board`'s fields are already plain tuples of plain ints.

Frontend contract: connect with `EventSource`, redraw the board on every
`onmessage`, treat `done: true` as "this episode ended" (score resets, a new
`episode` value starts arriving) — the stream itself keeps running until
`n_episodes` is exhausted, then the connection closes naturally.

### Dependencies & deployment

New optional-dependency group in `pyproject.toml`:

```toml
[project.optional-dependencies]
render = ["pygame>=2.5"]
plot = ["matplotlib>=3.11.1"]
api = ["fastapi>=0.115", "uvicorn[standard]>=0.32", "httpx>=0.27"]
```

`httpx` is included because `fastapi.testclient.TestClient` requires it.

`Procfile` (repo root):

```
web: uv run --extra api uvicorn api:app --host 0.0.0.0 --port $PORT
```

Railway sets `$PORT`; no other environment configuration is required since
CORS is wide open and the Q-table path is fixed.

## Testing impact

New `tests/test_api.py` using `fastapi.testclient.TestClient`:

- `/train` and `/play` with small, fast params (`n_episodes=1`,
  `grid_size=8`, high `fps` to keep `asyncio.sleep` calls negligible) assert
  `200`, `content-type` starts with `text/event-stream`, the body parses as
  one-or-more `data: {...}` frames, and each frame has the expected keys.
- Out-of-range `n_episodes`/`grid_size` on both endpoints assert `422`.
- `/train`'s stream includes a non-null `epsilon`; `/play`'s is always `null`.

`tests/test_train.py::test_saves_q_table_to_path` moves to `tests/test_main.py`
(renamed to something like `test_train_command_saves_q_table_to_path`) since
saving is now `main.py`'s responsibility, not `train()`'s. `test_train.py`
gains no replacement assertion beyond confirming `train()` no longer writes
any file (or simply loses the test — `train()` not touching the filesystem
is the default expectation for a generator, not a behavior worth pinning on
its own).

No test coverage is added for `Procfile` itself or for actually deploying to
Railway — that's operational, not something `pytest` can verify.
