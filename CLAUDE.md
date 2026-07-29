# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A tabular Q-learning agent that learns to play Snake. Still headless — no rendering (a pygame renderer is planned as future work) — but `main.py` now exposes a small CLI. `train.py` runs episodes against `SnakeEnv` and dumps a learned Q-table to `q_table.json`; `play.py` loads a saved Q-table and runs the agent greedily (`epsilon=0`), printing scores with no learning/updates.

## Commands

Package manager is `uv` (`uv.lock` present, Python `>=3.13` per `.python-version`/`pyproject.toml`).

```bash
uv sync                                    # install deps
uv run python main.py train                # train with all defaults (30,000 episodes, 20x20 grid)
uv run python main.py train --n-episodes 5000 --grid-size 10 --save-path other.json
uv run python main.py play                 # load q_table.json, play 100 episodes greedily (epsilon=0)
uv run python main.py play --n-episodes 10 --q-table-path other.json

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
                      loads a saved table and runs greedily, no learning)
      ↑
  main.py       (argparse CLI: `train`/`play` subcommands, dispatches
                 straight to train() / play() with parsed overrides)
```

- **`Snake`** owns only `body` (deque), `pos_set` (kept in sync with body on every `move`), and `direction`. No grid/food/game-over knowledge.
- **`SnakeState`** (`snake_state.py`) is a frozen dataclass built via the classmethod `SnakeState.from_world(snake, food, grid_size)`. It doesn't hold a reference to the snake — it's a value snapshot with an `.index` property that maps directly to a row in the Q-table.
- **`SnakeEnv`** owns the actual game world (`grid_size`, live `Snake`, `food`, `steps_since_food`) and is the only thing that mutates game state. `step(action) -> StepResult` where `StepResult` is a frozen dataclass (`state, reward, done, truncated, info`) — this replaced an earlier plain-tuple return specifically to kill fragile positional unpacking (see `docs/superpowers/specs/2026-07-27-q-learning-agent-design.md`).
- **`QLearningAgent`** is deliberately decoupled from everything above — it only needs `SnakeState.N_STATES` at construction and thereafter only sees integer state indices and `Action` values. Q-table is a plain `list[list[float]]`, persisted as unversioned JSON (`save`/`load`, path-typed via `pathlib.Path`) with no shape validation against `n_states`/`n_actions`.
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

**`train()`'s periodic log line reports an all-time top score** alongside the rolling `avg_score`: a running `max()` over every episode's `info["score"]` since training started (not windowed like `avg_score`), so it's monotonically non-decreasing across a run.

**Collision/tail-exclusion logic is duplicated, not shared**, between `SnakeEnv.step` (excludes tail from the obstacle set _unless_ food was just consumed this move, since then the tail doesn't vacate) and `SnakeState.from_world`'s danger ray-casting (unconditionally excludes the tail). They happen to agree in practice but could drift if one side changes without the other — check both when touching collision or danger-sensing logic.

**Known spec/code discrepancy**: `docs/superpowers/specs/2026-07-27-distance-bucketed-state-design.md` calls for bumping `QLearningAgent.epsilon_decay_episodes` to 100,000 and `train()`'s `n_episodes` to 200,000 to match the 22x larger state space (1600 vs. 72 states). Current code still defaults to `epsilon_decay_episodes=5_000` (`q_agent.py`) and `n_episodes=30_000` (`train.py`) — these were deliberately reverted (see commit `17a4a3e`, "revert training defaults"), and tests assert the smaller values. Don't "fix" these back to the spec's numbers without checking why they were reverted first.

**`q_table.json` is gitignored** and has no versioning — retraining from scratch is the expected workflow after any state-space or reward change, not migrating an old table.

## Design docs

`docs/superpowers/specs/` and `docs/superpowers/plans/` hold spec-then-plan documents (from the "superpowers" spec-driven-development workflow) for each major change: the original env design, the Q-learning agent + `StepResult` refactor, and the distance-bucketed state redesign. Specs are the useful ones for understanding _why_; plans are step-by-step implementation breakdowns of the same work.

## Testing conventions

Plain `pytest` (no `unittest.TestCase`), organized as `Test*` classes grouping methods by behavior under test, no `conftest.py`, no `@pytest.fixture` — setup uses small helper functions instead (e.g. `make_state()`, `make_snake()` in `test_snake_state.py`). Heavy use of `@pytest.mark.parametrize`, including class-level parametrization (e.g. `TestFromWorldFoodBuckets` runs every test across all 4 `Direction`s).

`tests/test_snake_env.py::TestLifecycle::test_random_policy_episodes_hold_invariants_every_step` is a 500-episode soak test (`random.seed(1)`) that checks invariants every step — this is what previously caught a `pos_set` desync bug in `Snake.move`. Worth rerunning specifically after any change to `Snake`, `SnakeState`, or `SnakeEnv`.
