# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A tabular Q-learning agent that learns to play Snake. `train.py` and `play.py` are generators that run episodes against `SnakeEnv` and yield `EpisodeStep` objects — neither prints anything itself. `main.py` is the CLI that consumes those generators and does the printing: `train` runs episodes and dumps a learned Q-table to `q_table.json`; `play` loads a saved Q-table and runs the agent greedily (`epsilon=0`), with no learning/updates. `watch.py` is a separate, optional entry point that renders either a live training run or a saved agent playing, via `renderer.py`'s `PygameRenderer` — pygame is an optional dependency (`uv sync --extra render`), never required for headless training/play. `main.py`'s `train`/`play` subcommands also take an optional `--plot`/`--plot-path`, rendering via `plotting.py` (matplotlib, another optional dependency — `uv sync --extra plot`) — training progress (rolling avg score) or the play-run score distribution, respectively. A third optional entry point, `api.py` (FastAPI, `uv sync --extra api`), exposes `train()`/`play()` over HTTP as Server-Sent Events (`GET /train`, `GET /play`) for a browser frontend — deployable to Railway via the included `Procfile`. `train()` itself no longer writes `q_table.json`; that save now happens in `main.py`'s `_run_train` after the generator is exhausted, so `api.py` can stream a training run without touching disk. `/play` always loads the committed `example_q_table.json` rather than the gitignored `q_table.json`, since a fresh deployment has no trained table of its own.

## Commands

Package manager is `uv` (`uv.lock` present, Python `>=3.13` per `.python-version`/`pyproject.toml`).

```bash
uv sync                                    # install deps
uv run python main.py train                # train with all defaults (30,000 episodes, 20x20 grid)
uv run python main.py train --n-episodes 5000 --grid-size 10 --save-path other.json
uv run python main.py train --alpha 0.05 --gamma 0.95 --epsilon-decay-episodes 10000  # override AgentConfig hyperparameters
uv run python main.py play                 # load q_table.json, play 100 episodes greedily (epsilon=0)
uv run python main.py play --n-episodes 10 --q-table-path other.json

uv sync --extra render                     # install optional pygame dependency
uv run --extra render python watch.py play --q-table-path q_table.json
uv run --extra render python watch.py train --render-every 500  # render-every is training-only; play renders every episode

uv sync --extra plot                       # install optional matplotlib dependency
uv run --extra plot python main.py train --plot --plot-path training.png  # plot rolling avg score over training; omit --plot-path to show interactively
uv run --extra plot python main.py play --plot --plot-path scores.png     # plot the score distribution across the play run

uv sync --extra api                        # install optional fastapi/uvicorn dependencies
uv run --extra api uvicorn api:app --reload  # serve /train and /play as SSE streams on :8000
uv run --extra api pytest tests/test_api.py  # api-specific tests (needs httpx from the api extra)

uv run pytest                              # full test suite
uv run pytest tests/test_snake_env.py      # one file
uv run pytest tests/test_q_agent.py::TestUpdate                              # one class
uv run pytest tests/test_q_agent.py::TestUpdate::test_real_death_does_not_bootstrap  # one test
```

There is no linter/formatter configured (`.ruff_cache` present but no `[tool.ruff]` in `pyproject.toml` and ruff isn't a declared dependency — it's leftover from an ad hoc `uvx ruff` run, not a project convention). `uvx ruff check .` works with defaults if you want to lint, but there's no established rule set to match.

## Architecture

Strict layering, each module only knowing about the one below it:

```
snake_types.py  (Action, Direction — pure value types)
      ↑
   snake.py     (Snake — body/heading geometry only)
      ↑
snake_state.py  (SnakeState — discretizes Snake+food into an RL state)
      ↑
 snake_env.py   (SnakeEnv — Gym-like game loop: reset()/step())
      ↑
  q_agent.py    (QLearningAgent — knows nothing about Snake/Env, only
                 works with SnakeState.index ints and Action)
      ↑
train.py / play.py  (glue: train() runs episodes and learns; play()
                      loads a saved table and runs greedily, no learning —
                      both are generators yielding EpisodeStep, no printing)
      ↑
episode_step.py (EpisodeStep — frozen dataclass wrapping StepResult +
                 the agent, yielded by both train() and play())
      ↑
  main.py       (argparse CLI: `train`/`play` subcommands, consumes the
                 EpisodeStep generators and does all the printing)
```

A bottom-layer module, `config.py`, sits alongside `snake_types.py` — frozen dataclasses (`AgentConfig`, `RenderConfig`, `TrainConfig`, `PlayConfig`) that any layer imports for its own parameters, replacing what used to be scalar kwargs duplicated across `main.py`/`watch.py`/`train.py`/`play.py`/`q_agent.py`/`renderer.py`. `QLearningAgent` takes an `AgentConfig`, `PygameRenderer` takes a `RenderConfig`, and `train()`/`play()` take a `TrainConfig`/`PlayConfig` (`TrainConfig` itself carries an `AgentConfig`). See `docs/superpowers/specs/2026-07-29-centralized-config-design.md`.

Off to the side, a separate optional branch renders gameplay: `renderer.py` (`PygameRenderer`, knows only about `SnakeEnv`'s `Board`) is driven by `watch.py`, which calls `train()`/`play()` directly and feeds each yielded step's board to the renderer. Neither `main.py` nor `train.py`/`play.py` import pygame or `watch.py`.

Another optional, off-to-the-side branch plots results: `plotting.py` (`plot_training_progress`, `plot_score_distribution` — matplotlib, `uv sync --extra plot`) is called from `main.py`'s `_run_train`/`_run_play` only when `--plot` is passed, via a *lazy* import inside the `if plot:` branch — this keeps `main.py` itself importable without matplotlib installed, the same way `train.py`/`play.py` stay importable without pygame. `_run_train` reuses the same 500-episode rolling-average checkpoints it already computes for the printed progress line; `_run_play` reuses the same per-episode `scores` list it already prints and averages. Like `renderer.py`, `plotting.py` has no automated test coverage (verified manually with `MPLBACKEND=Agg` instead) — same reasoning: the dependency is optional and the output is visual.

A third optional, off-to-the-side branch serves results over HTTP: `api.py` (FastAPI, `uv sync --extra api`) exposes `train()`/`play()` as Server-Sent Events (`GET /train`, `GET /play`) for a browser frontend, deployable to Railway via the included `Procfile`. Like `renderer.py`/`plotting.py`, `api.py` is imported by nothing else in the codebase. `/play` always loads the committed `example_q_table.json` rather than the gitignored `q_table.json`, since a fresh deployment has no trained table of its own — `QLearningAgent.load` does no shape validation, so if `SnakeState.N_STATES` ever changes (e.g. `MAX_DANGER_SCAN` or the food-bucket ranges), `example_q_table.json` must be re-copied from a freshly trained `q_table.json` or the deployed `/play` endpoint will `IndexError` mid-stream.

- **`Snake`** owns only `body` (deque), `pos_set` (kept in sync with body on every `move`), and `direction`. No grid/food/game-over knowledge.
- **`SnakeState`** (`snake_state.py`) is a frozen dataclass built via the classmethod `SnakeState.from_world(snake, food, grid_size)`. It doesn't hold a reference to the snake — it's a value snapshot with an `.index` property that maps directly to a row in the Q-table.
- **`SnakeEnv`** owns the actual game world (`grid_size`, live `Snake`, `food`, `steps_since_food`) and is the only thing that mutates game state. `step(action) -> StepResult` where `StepResult` is a frozen dataclass (`state, reward, done, truncated, info, board`) — this replaced an earlier plain-tuple return specifically to kill fragile positional unpacking (see `docs/superpowers/specs/2026-07-27-q-learning-agent-design.md`). `board: Board` (added later, see `docs/superpowers/specs/2026-07-29-pygame-renderer-design.md`) is a raw-coordinate snapshot (`grid_size`, `snake_body`, `food`) for `watch.py`'s renderer — distinct from `state: SnakeState`, which is a lossy RL encoding that can't reconstruct pixel positions. `SnakeEnv.render_state()` builds it.
- **`QLearningAgent`** is deliberately decoupled from everything above — it only needs `SnakeState.N_STATES` and an `AgentConfig` (`config.py`) at construction and thereafter only sees integer state indices and `Action` values. Q-table is a plain `list[list[float]]`, persisted as unversioned JSON (`save`/`load`, path-typed via `pathlib.Path`) with no shape validation against `n_states`/`n_actions`.
- **`play.py`** mirrors `train.py`'s episode loop but forces `agent.epsilon = 0.0` after loading and never calls `agent.update` — pure greedy inference, no learning. It fails fast with a clear `FileNotFoundError` if the Q-table path doesn't exist, rather than surfacing `json.load`'s raw traceback.

`train.py`, `play.py`, and `main.py` never touch `Snake` or grid geometry directly — only `SnakeEnv` and `SnakeState`.

## Key design decisions (non-obvious from a single file)

**State encoding is rotation-invariant.** `SnakeState` has 5 discretized fields: 3 "danger" ray-cast distances (straight/right/left _relative to current heading_, not absolute grid direction) and 2 food-position components (forward/lateral, also heading-relative). This means the agent doesn't need a separate policy per absolute facing.

- Danger rays (`_ray_distance`, `snake_state.py`) scan up to `MAX_DANGER_SCAN=6` cells and bucket the result via `_danger_bucket` into 4 buckets: `0`=immediate collision, `1`=dist 1-2, `2`=dist 3-5, `3`=clear. Occupied cells exclude the snake's own tail (it vacates on a normal move).
- Food position is bucketed via `_food_bucket` into `{-2,-1,0,1,2}` per axis (near vs. far, signed).
- `SnakeState.index` packs all 5 fields into one int via mixed-radix encoding (base 4 for danger, base 5 for food) → `N_STATES = 4**3 * 5**2 = 1600`. This is the only coupling between `SnakeState` and `QLearningAgent`'s table size — `train.py` passes `SnakeState.N_STATES` into the agent constructor; there's no runtime check that a saved `q_table.json` actually has this shape.
- This replaced an earlier boolean-danger/3-way-sign encoding (72 states) that plateaued around avg score ~21-22 and couldn't distinguish near vs. far obstacles. See `docs/superpowers/specs/2026-07-27-distance-bucketed-state-design.md` for the full rationale; `docs/superpowers/specs/2026-07-26-snake-environment-design.md` covers the original env design.

**`done` vs `truncated` controls Bellman bootstrapping — easy to get backwards.** `done=True` covers both real death (wall/self collision) and starvation truncation (`steps_since_food > 100 * snake.length`). `QLearningAgent.update` only skips bootstrapping (`target = reward` alone) when `done and not truncated` — i.e. real death. On truncation, or on any normal step, it still bootstraps via `reward + gamma * max(q_table[next_index])`, because a "starved but otherwise alive" state shouldn't be taught it's worthless the way real death should be. Get this inverted and the agent will systematically under- or over-value near-timeout states.

**Reward is pure and sparse**: `FOOD_REWARD=10`, `DEATH_REWARD=-10`, `STEP_REWARD=0`. No distance-based shaping.

**`grid_size` defaults to 20** (`SnakeEnv.__init__`, `train()`, `play()`) — bumped up from an original 12x12, closer to a classic arcade-size board. This doesn't change `N_STATES` (still 1600 — driven by `MAX_DANGER_SCAN` and food buckets, not grid size), so `n_episodes` defaults were deliberately left alone (see the reverted-defaults note below) even though a bigger board means food/danger are relatively farther away more often.

**`main.py`'s `_run_train` periodic log line reports an all-time top score** alongside the rolling `avg_score`, reading both off the `EpisodeStep` stream `train()` yields: a running `max()` over every episode's `info["score"]` since training started (not windowed like `avg_score`), so it's monotonically non-decreasing across a run.

**Collision/tail-exclusion logic is duplicated, not shared**, between `SnakeEnv.step` (excludes tail from the obstacle set _unless_ food was just consumed this move, since then the tail doesn't vacate) and `SnakeState.from_world`'s danger ray-casting (unconditionally excludes the tail). They happen to agree in practice but could drift if one side changes without the other — check both when touching collision or danger-sensing logic.

**Known spec/code discrepancy**: `docs/superpowers/specs/2026-07-27-distance-bucketed-state-design.md` calls for bumping `QLearningAgent.epsilon_decay_episodes` to 100,000 and `train()`'s `n_episodes` to 200,000 to match the 22x larger state space (1600 vs. 72 states). Current code still defaults to `AgentConfig.epsilon_decay_episodes=5_000` and `TrainConfig.n_episodes=30_000` (both in `config.py` — see `docs/superpowers/specs/2026-07-29-centralized-config-design.md`) — these were deliberately reverted (see commit `17a4a3e`, "revert training defaults"), and tests assert the smaller values. Don't "fix" these back to the spec's numbers without checking why they were reverted first.

**`q_table.json` is gitignored** and has no versioning — retraining from scratch is the expected workflow after any state-space or reward change, not migrating an old table.

**`Board.snake_body` is a real copy, not a live reference into `Snake.body`.** `SnakeEnv.render_state()` does `tuple(self.snake.body)` specifically so a yielded `Board` stays frozen even after the next `step()` mutates the snake's deque in place — otherwise every consumer of `train()`/`play()`'s `EpisodeStep` stream (`watch.py` in particular) would silently see the *current* snake instead of the one from the step it was handed. `tests/test_snake_env.py::TestStepBoard::test_board_is_a_snapshot_not_a_live_view` pins this.

## Design docs

`docs/superpowers/specs/` and `docs/superpowers/plans/` hold spec-then-plan documents (from the "superpowers" spec-driven-development workflow) for each major change: the original env design, the Q-learning agent + `StepResult` refactor, the distance-bucketed state redesign, and the pygame renderer (`Board`/`EpisodeStep`, the `train()`/`play()` generator conversion, `renderer.py`/`watch.py`). Specs are the useful ones for understanding _why_; plans are step-by-step implementation breakdowns of the same work.

## Testing conventions

Plain `pytest` (no `unittest.TestCase`), organized as `Test*` classes grouping methods by behavior under test, no `conftest.py`, no `@pytest.fixture` — setup uses small helper functions instead (e.g. `make_state()`, `make_snake()` in `test_snake_state.py`). Heavy use of `@pytest.mark.parametrize`, including class-level parametrization (e.g. `TestFromWorldFoodBuckets` runs every test across all 4 `Direction`s).

`tests/test_snake_env.py::TestLifecycle::test_random_policy_episodes_hold_invariants_every_step` is a 500-episode soak test (`random.seed(1)`) that checks invariants every step — this is what previously caught a `pos_set` desync bug in `Snake.move`. Worth rerunning specifically after any change to `Snake`, `SnakeState`, or `SnakeEnv`.
