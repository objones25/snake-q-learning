# Pygame Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pygame renderer that can watch a live training run or a loaded agent playing greedily, without `train.py`/`play.py` depending on pygame or on rendering at all.

**Architecture:** `SnakeEnv.step()` starts returning an immutable `Board` snapshot (raw snake body + food + grid size) as part of `StepResult`. `train()`/`play()` become generators yielding one `EpisodeStep` (episode index, `StepResult`, agent) per step, with printing removed entirely. `main.py` becomes a consumer of that stream that prints; a new `watch.py` becomes a second consumer of the same stream that renders via a new `PygameRenderer` (in `renderer.py`). Neither consumer duplicates the other's — or `train`'s/`play`'s — loop logic.

**Tech Stack:** Python 3.13, pytest, pygame (optional extra, via `uv`).

## Global Constraints

- pygame is an **optional** dependency group (`[project.optional-dependencies] render = ["pygame>=2.5"]`) — never added to `dependencies`. A plain `uv sync` (no `--extra render`) must continue to work and must never require pygame to be importable.
- No changes to collision detection, reward logic, `reset()`, or any other `SnakeEnv` game-rule behavior — this plan is purely additive.
- No pause/rewind/speed-slider UI in the renderer, and no terminal/ASCII renderer or other backend — pygame only, draw-and-quit-on-close.
- Renderer colors are fixed module-level constants, not configurable: `BG_COLOR = (0, 0, 0)`, `SNAKE_COLOR = (0, 200, 0)`, `FOOD_COLOR = (200, 0, 0)`.
- Renderer defaults: `cell_size=24`, `fps=15`.
- `render_every` sampling exists only on `watch_train`, not `watch_play`.
- Spec of record: `docs/superpowers/specs/2026-07-29-pygame-renderer-design.md`.

---

### Task 1: `Board` snapshot on `StepResult`

**Files:**

- Modify: `snake_env.py` (currently 87 lines)
- Test: `tests/test_snake_env.py`

**Interfaces:**

- Produces: `Board` dataclass (`grid_size: int, snake_body: tuple[tuple[int, int], ...], food: tuple[int, int]`) in `snake_env.py`; `StepResult.board: Board` (new 6th field); `SnakeEnv.render_state() -> Board`.

- [ ] **Step 1: Write the failing tests**

Add this new test class to `tests/test_snake_env.py`, right after `class TestStepTruncatedFlag` (i.e. before `class TestLifecycle`):

```python
class TestStepBoard:
    def test_board_reflects_snake_body_and_food_after_normal_step(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        result = env.step(Action.STRAIGHT)
        assert result.board.grid_size == 12
        assert result.board.snake_body == tuple(env.snake.body)
        assert result.board.food == env.food

    def test_board_reflects_snake_body_on_collision(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.snake = Snake((11, 5), Direction.RIGHT)
        env.food = (0, 0)
        result = env.step(Action.STRAIGHT)
        assert result.done is True
        assert result.board.snake_body == tuple(env.snake.body)
        assert result.board.food == (0, 0)

    def test_render_state_matches_step_board(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        result = env.step(Action.STRAIGHT)
        assert env.render_state() == result.board
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_snake_env.py::TestStepBoard -v`
Expected: FAIL — `AttributeError: 'StepResult' object has no attribute 'board'` (and `AttributeError: 'SnakeEnv' object has no attribute 'render_state'` for the third test).

- [ ] **Step 3: Implement `Board`, `StepResult.board`, and `SnakeEnv.render_state()`**

Replace the full contents of `snake_env.py` with:

```python
import random
from dataclasses import dataclass

from snake import Snake
from snake_state import SnakeState
from snake_types import Action, Direction

FOOD_REWARD = 10
DEATH_REWARD = -10
STEP_REWARD = 0


@dataclass(frozen=True, slots=True)
class Board:
    grid_size: int
    snake_body: tuple[tuple[int, int], ...]
    food: tuple[int, int]


@dataclass(frozen=True, slots=True)
class StepResult:
    state: SnakeState
    reward: float
    done: bool
    truncated: bool
    info: dict
    board: Board


class SnakeEnv:
    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.snake: Snake
        self.food: tuple[int, int]
        self.steps_since_food = 0
        self._all_cells = {
            (x, y) for x in range(grid_size) for y in range(grid_size)
        }

    def reset(self) -> SnakeState:
        start_pos = (self.grid_size // 2, self.grid_size // 2)
        direction = random.choice(list(Direction))
        self.snake = Snake(start_pos, direction)
        self.food = self._place_food()
        self.steps_since_food = 0
        return SnakeState.from_world(self.snake, self.food, self.grid_size)

    def _place_food(self) -> tuple[int, int]:
        free_cells = list(self._all_cells - self.snake.pos_set)
        return random.choice(free_cells)

    def render_state(self) -> Board:
        return Board(
            grid_size=self.grid_size,
            snake_body=tuple(self.snake.body),
            food=self.food,
        )

    def step(self, action: Action) -> StepResult:
        """Advance the environment by one action.

        `done=True` can mean either termination (wall/self collision) or
        truncation (starvation timeout) — check `truncated` to distinguish
        them, since a training loop typically bootstraps the value estimate
        through truncation but not through real termination.
        """
        if action == Action.RIGHT:
            self.snake.turn_right()
        elif action == Action.LEFT:
            self.snake.turn_left()

        new_head = self.snake.direction.apply(self.snake.head)
        food_consumed = new_head == self.food

        out_of_bounds = not (
            0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size
        )
        occupied = (
            self.snake.pos_set
            if food_consumed
            else self.snake.pos_set - {self.snake.tail}
        )
        collision = out_of_bounds or new_head in occupied

        if collision:
            state = SnakeState.from_world(self.snake, self.food, self.grid_size)
            return StepResult(
                state, DEATH_REWARD, True, False,
                {"score": self.snake.length}, self.render_state(),
            )

        self.snake.move(food_consumed)

        if food_consumed:
            reward = FOOD_REWARD
            self.steps_since_food = 0
            self.food = self._place_food()
        else:
            reward = STEP_REWARD
            self.steps_since_food += 1

        done = self.steps_since_food > 100 * self.snake.length
        state = SnakeState.from_world(self.snake, self.food, self.grid_size)
        return StepResult(
            state=state, reward=reward, done=done, truncated=done,
            info={"score": self.snake.length}, board=self.render_state(),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_snake_env.py -v`
Expected: PASS — all tests, including the three new ones and every pre-existing test in the file (none construct `StepResult` positionally, so none break).

- [ ] **Step 5: Commit**

```bash
git add snake_env.py tests/test_snake_env.py
git commit -m "$(cat <<'EOF'
Add Board snapshot to StepResult for the future renderer

StepResult.board carries the raw snake body/food/grid_size a renderer
needs to draw a frame, distinct from SnakeState's lossy RL encoding.
SnakeEnv.render_state() produces it; collision logic is unchanged.
EOF
)"
```

---

### Task 2: `EpisodeStep`

**Files:**

- Create: `episode_step.py`
- Test: `tests/test_episode_step.py`

**Interfaces:**

- Consumes: `StepResult` (`snake_env.py`, Task 1), `QLearningAgent` (`q_agent.py`).
- Produces: `EpisodeStep` dataclass (`episode: int, result: StepResult, agent: QLearningAgent`) in `episode_step.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_episode_step.py`:

```python
from episode_step import EpisodeStep
from q_agent import QLearningAgent
from snake_env import Board, StepResult
from snake_state import SnakeState


class TestEpisodeStep:
    def test_round_trips_its_fields(self):
        board = Board(grid_size=8, snake_body=((1, 1),), food=(2, 2))
        state = SnakeState(dng_straight=3, dng_right=3, dng_left=3, food_fwd=0, food_lat=0)
        result = StepResult(
            state=state, reward=0.0, done=False, truncated=False,
            info={"score": 1}, board=board,
        )
        agent = QLearningAgent(n_states=SnakeState.N_STATES)

        step = EpisodeStep(episode=5, result=result, agent=agent)

        assert step.episode == 5
        assert step.result is result
        assert step.agent is agent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_episode_step.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'episode_step'`.

- [ ] **Step 3: Implement `EpisodeStep`**

Create `episode_step.py`:

```python
from dataclasses import dataclass

from q_agent import QLearningAgent
from snake_env import StepResult


@dataclass(frozen=True, slots=True)
class EpisodeStep:
    episode: int
    result: StepResult
    agent: QLearningAgent
```

This lives in its own module rather than `snake_env.py` because embedding `QLearningAgent` there would make the env layer import the agent layer, inverting this project's bottom-up layering (`snake_types → snake → snake_state → snake_env → q_agent → train/play`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_episode_step.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add episode_step.py tests/test_episode_step.py
git commit -m "$(cat <<'EOF'
Add EpisodeStep, the per-step event train()/play() will yield

Bundles episode index, StepResult, and the live agent so both a CLI
printer and a future renderer can consume the same stream without
either needing SnakeEnv directly.
EOF
)"
```

---

### Task 3: Convert `train()` into a generator

**Files:**

- Modify: `train.py` (currently 51 lines)
- Test: `tests/test_train.py`

**Interfaces:**

- Consumes: `EpisodeStep` (Task 2).
- Produces: `train(n_episodes=30_000, grid_size=20, save_path=Path("q_table.json")) -> Iterator[EpisodeStep]`. No longer prints; no longer returns a value (falls off the end after `agent.save(save_path)`).

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `tests/test_train.py` with:

```python
import inspect
import tempfile
from pathlib import Path

import pytest

from q_agent import QLearningAgent
from snake_state import SnakeState
from train import train


class TestTrain:
    def test_returns_agent_with_correctly_shaped_q_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q_table.json"
            steps = list(train(n_episodes=20, grid_size=8, save_path=path))
            agent = steps[-1].agent

        assert isinstance(agent, QLearningAgent)
        assert len(agent.q_table) == SnakeState.N_STATES
        assert all(len(row) == 3 for row in agent.q_table)

    def test_epsilon_decreases_from_start_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q_table.json"
            steps = list(train(n_episodes=20, grid_size=8, save_path=path))
            agent = steps[-1].agent

        assert agent.epsilon == pytest.approx(0.996238)

    def test_saves_q_table_to_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q_table.json"
            list(train(n_episodes=20, grid_size=8, save_path=path))
            assert path.exists()

    def test_death_is_passed_to_update_as_not_truncated(self, monkeypatch):
        seen = []
        original = QLearningAgent.update

        def spy(self, state_index, action, reward, next_index, done, truncated):
            if done:
                seen.append((done, truncated))
            original(self, state_index, action, reward, next_index, done, truncated)

        monkeypatch.setattr(QLearningAgent, "update", spy)
        with tempfile.TemporaryDirectory() as tmpdir:
            list(train(n_episodes=20, grid_size=8, save_path=Path(tmpdir) / "q.json"))

        assert (True, False) in seen


class TestDefaults:
    def test_default_n_episodes_is_30000(self):
        assert inspect.signature(train).parameters["n_episodes"].default == 30_000

    def test_default_grid_size_is_20(self):
        assert inspect.signature(train).parameters["grid_size"].default == 20

    def test_save_path_is_path_typed_with_q_table_json_default(self):
        param = inspect.signature(train).parameters["save_path"]
        assert param.annotation is Path
        assert param.default == Path("q_table.json")
```

Note: `test_logs_top_score_alongside_avg_score` is intentionally removed here — `train()` no longer prints; that coverage moves to `tests/test_main.py` in Task 5.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL — e.g. `steps[-1].agent` raises `AttributeError` because `train()` still returns a plain `QLearningAgent` (not yet a generator of `EpisodeStep`), so `list(train(...))` currently produces a `TypeError: 'QLearningAgent' object is not iterable` (since `train()` isn't a generator yet, calling it runs to completion and returns an `agent`, which `list()` then tries to iterate).

- [ ] **Step 3: Convert `train()` to a generator**

Replace the full contents of `train.py` with:

```python
from collections.abc import Iterator
from pathlib import Path

from episode_step import EpisodeStep
from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState


def train(
    n_episodes: int = 30_000,
    grid_size: int = 20,
    save_path: Path = Path("q_table.json"),
) -> Iterator[EpisodeStep]:
    env = SnakeEnv(grid_size=grid_size)
    agent = QLearningAgent(n_states=SnakeState.N_STATES)

    for episode in range(n_episodes):
        agent.set_epsilon_for_episode(episode)
        state = env.reset()
        result = None
        while result is None or not result.done:
            action = agent.choose_action(state.index)
            result = env.step(action)
            agent.update(
                state.index,
                action,
                result.reward,
                result.state.index,
                result.done,
                result.truncated,
            )
            state = result.state
            yield EpisodeStep(episode=episode, result=result, agent=agent)

    agent.save(save_path)


if __name__ == "__main__":
    for _ in train():
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "$(cat <<'EOF'
Convert train() into a generator yielding EpisodeStep

Removes all printing from train.py; callers now iterate the stream of
EpisodeSteps themselves. This lets a future CLI printer and a future
renderer consume the exact same loop without either duplicating it.
EOF
)"
```

---

### Task 4: Convert `play()` into a generator

**Files:**

- Modify: `play.py` (currently 43 lines)
- Test: `tests/test_play.py`

**Interfaces:**

- Consumes: `EpisodeStep` (Task 2).
- Produces: `play(n_episodes=100, grid_size=20, q_table_path=Path("q_table.json")) -> Iterator[EpisodeStep]`. No longer prints; no longer returns a value. Still raises `FileNotFoundError` immediately if `q_table_path` doesn't exist — but (generator gotcha) only once the caller starts iterating, not on the bare call.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `tests/test_play.py` with:

```python
import inspect
import tempfile
from pathlib import Path

import pytest

from play import play
from q_agent import QLearningAgent
from snake_state import SnakeState


class TestMissingQTable:
    def test_raises_clear_error_when_q_table_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "does_not_exist.json"
            with pytest.raises(FileNotFoundError, match=str(missing_path)):
                list(play(n_episodes=1, grid_size=8, q_table_path=missing_path))


class TestPlay:
    def _make_q_table(self, tmpdir):
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        path = Path(tmpdir) / "q_table.json"
        agent.save(path)
        return path

    def test_runs_n_episodes_and_returns_a_score_per_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_q_table(tmpdir)
            steps = list(play(n_episodes=5, grid_size=8, q_table_path=path))

        scores = [step.result.info["score"] for step in steps if step.result.done]
        assert len(scores) == 5
        assert all(isinstance(score, int) and score >= 1 for score in scores)

    def test_forces_epsilon_to_zero(self, monkeypatch):
        seen_epsilons = []
        original = QLearningAgent.choose_action

        def spy(self, state_index):
            seen_epsilons.append(self.epsilon)
            return original(self, state_index)

        monkeypatch.setattr(QLearningAgent, "choose_action", spy)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_q_table(tmpdir)
            list(play(n_episodes=1, grid_size=8, q_table_path=path))

        assert seen_epsilons
        assert all(epsilon == 0.0 for epsilon in seen_epsilons)


class TestDefaults:
    def test_default_n_episodes_is_100(self):
        assert inspect.signature(play).parameters["n_episodes"].default == 100

    def test_default_grid_size_is_20(self):
        assert inspect.signature(play).parameters["grid_size"].default == 20

    def test_default_q_table_path_is_path_typed(self):
        param = inspect.signature(play).parameters["q_table_path"]
        assert param.annotation is Path
        assert param.default == Path("q_table.json")
```

Note: `test_prints_per_episode_scores_and_summary` is intentionally removed — `play()` no longer prints; that coverage moves to `tests/test_main.py` in Task 5.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_play.py -v`
Expected: FAIL — `test_raises_clear_error_when_q_table_missing` fails because calling `play(...)` (not yet a generator) already raises before `list(...)` wraps it, which still passes actually — but `test_runs_n_episodes_and_returns_a_score_per_episode` fails with `AttributeError: 'int' object has no attribute 'result'` (or similar), since `play()` currently returns a plain `list[int]` of scores, not an iterable of `EpisodeStep`.

- [ ] **Step 3: Convert `play()` to a generator**

Replace the full contents of `play.py` with:

```python
from collections.abc import Iterator
from pathlib import Path

from episode_step import EpisodeStep
from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState


def play(
    n_episodes: int = 100,
    grid_size: int = 20,
    q_table_path: Path = Path("q_table.json"),
) -> Iterator[EpisodeStep]:
    if not q_table_path.exists():
        raise FileNotFoundError(
            f"No q_table found at {q_table_path} — run `main.py train` first"
        )

    env = SnakeEnv(grid_size=grid_size)
    agent = QLearningAgent(n_states=SnakeState.N_STATES)
    agent.load(q_table_path)
    agent.epsilon = 0.0

    for episode in range(n_episodes):
        state = env.reset()
        result = None
        while result is None or not result.done:
            action = agent.choose_action(state.index)
            result = env.step(action)
            state = result.state
            yield EpisodeStep(episode=episode, result=result, agent=agent)


if __name__ == "__main__":
    for _ in play():
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_play.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add play.py tests/test_play.py
git commit -m "$(cat <<'EOF'
Convert play() into a generator yielding EpisodeStep

Mirrors the train() conversion: no more printing inside play.py, and
callers (main.py, and eventually watch.py) consume the same stream.
EOF
)"
```

---

### Task 5: `main.py` consumes the generators and owns printing

**Files:**

- Modify: `main.py` (currently 35 lines)
- Test: `tests/test_main.py`

**Interfaces:**

- Consumes: `train()`, `play()` (Tasks 3–4), both now `Iterator[EpisodeStep]`.
- Produces: `main.main(argv=None) -> None` (unchanged signature/behavior from the CLI's point of view — same flags, same stdout format).

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `tests/test_main.py` with:

```python
from pathlib import Path

import pytest

import main
from q_agent import QLearningAgent
from snake_state import SnakeState


class TestTrainDispatch:
    def test_train_subcommand_calls_train_with_defaults(self, monkeypatch):
        seen_kwargs = {}

        def fake_train(**kwargs):
            seen_kwargs.update(kwargs)
            return iter(())

        monkeypatch.setattr(main, "train", fake_train)

        main.main(["train"])

        assert seen_kwargs == {
            "n_episodes": 30_000,
            "grid_size": 20,
            "save_path": Path("q_table.json"),
        }

    def test_train_subcommand_honors_overrides(self, monkeypatch):
        seen_kwargs = {}

        def fake_train(**kwargs):
            seen_kwargs.update(kwargs)
            return iter(())

        monkeypatch.setattr(main, "train", fake_train)

        main.main(
            ["train", "--n-episodes", "500", "--grid-size", "10", "--save-path", "out.json"]
        )

        assert seen_kwargs == {
            "n_episodes": 500,
            "grid_size": 10,
            "save_path": Path("out.json"),
        }


class TestPlayDispatch:
    def test_play_subcommand_calls_play_with_defaults(self, monkeypatch):
        seen_kwargs = {}

        def fake_play(**kwargs):
            seen_kwargs.update(kwargs)
            return iter(())

        monkeypatch.setattr(main, "play", fake_play)

        main.main(["play"])

        assert seen_kwargs == {
            "n_episodes": 100,
            "grid_size": 20,
            "q_table_path": Path("q_table.json"),
        }

    def test_play_subcommand_honors_overrides(self, monkeypatch):
        seen_kwargs = {}

        def fake_play(**kwargs):
            seen_kwargs.update(kwargs)
            return iter(())

        monkeypatch.setattr(main, "play", fake_play)

        main.main(
            ["play", "--n-episodes", "5", "--grid-size", "10", "--q-table-path", "other.json"]
        )

        assert seen_kwargs == {
            "n_episodes": 5,
            "grid_size": 10,
            "q_table_path": Path("other.json"),
        }


class TestNoSubcommand:
    def test_missing_subcommand_exits_with_usage_error(self):
        with pytest.raises(SystemExit):
            main.main([])


class TestTrainAndPlayPrintProgress:
    def test_train_logs_top_score_alongside_avg_score(self, capsys, tmp_path):
        path = tmp_path / "q_table.json"
        main.main(["train", "--n-episodes", "20", "--grid-size", "8", "--save-path", str(path)])

        captured = capsys.readouterr()
        assert "top_score=" in captured.out

    def test_play_prints_per_episode_scores_and_summary(self, capsys, tmp_path):
        q_path = tmp_path / "q_table.json"
        QLearningAgent(n_states=SnakeState.N_STATES).save(q_path)

        main.main(["play", "--n-episodes", "3", "--grid-size", "8", "--q-table-path", str(q_path)])

        captured = capsys.readouterr()
        assert captured.out.count("episode") == 3
        assert "avg_score=" in captured.out
        assert "top_score=" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_main.py -v`
Expected: FAIL — the dispatch tests fail with something like `TypeError: 'NoneType' object is not iterable` (since `main.py` doesn't yet iterate `train()`/`play()`'s return value, and the fakes now return `iter(())` instead of `None`, but `main.py` isn't consuming it yet so calling the fake doesn't error — but the real dispatch code calling `train(...)` directly and discarding the result means these particular assertions on `seen_kwargs` still pass by coincidence). The two new `TestTrainAndPlayPrintProgress` tests fail clearly: `assert "top_score=" in captured.out` fails because `main.py` currently calls `train(...)`/`play(...)` and discards the generator without ever iterating it, so nothing runs and nothing prints.

- [ ] **Step 3: Implement `_run_train`/`_run_play` in `main.py`**

Replace the full contents of `main.py` with:

```python
import argparse
from collections import deque
from pathlib import Path

from play import play
from train import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--n-episodes", type=int, default=30_000)
    train_parser.add_argument("--grid-size", type=int, default=20)
    train_parser.add_argument("--save-path", type=Path, default=Path("q_table.json"))

    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--n-episodes", type=int, default=100)
    play_parser.add_argument("--grid-size", type=int, default=20)
    play_parser.add_argument("--q-table-path", type=Path, default=Path("q_table.json"))

    return parser


def _run_train(n_episodes: int, grid_size: int, save_path: Path) -> None:
    recent_scores: deque[int] = deque(maxlen=500)
    top_score = 0
    for step in train(n_episodes=n_episodes, grid_size=grid_size, save_path=save_path):
        if not step.result.done:
            continue
        recent_scores.append(step.result.info["score"])
        top_score = max(top_score, step.result.info["score"])
        if step.episode % 500 == 0:
            avg_score = sum(recent_scores) / len(recent_scores)
            print(
                f"episode {step.episode:6d}  epsilon={step.agent.epsilon:.3f}  "
                f"avg_score={avg_score:.2f}  top_score={top_score}"
            )


def _run_play(n_episodes: int, grid_size: int, q_table_path: Path) -> None:
    scores = []
    for step in play(n_episodes=n_episodes, grid_size=grid_size, q_table_path=q_table_path):
        if not step.result.done:
            continue
        score = step.result.info["score"]
        scores.append(score)
        print(f"episode {step.episode:6d}  score={score}")

    avg_score = sum(scores) / len(scores)
    print(f"avg_score={avg_score:.2f}  top_score={max(scores)}")


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.mode == "train":
        _run_train(n_episodes=args.n_episodes, grid_size=args.grid_size, save_path=args.save_path)
    elif args.mode == "play":
        _run_play(n_episodes=args.n_episodes, grid_size=args.grid_size, q_table_path=args.q_table_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_main.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full test suite to make sure nothing else regressed**

Run: `uv run pytest`
Expected: PASS — every test in `tests/`.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "$(cat <<'EOF'
Move CLI progress printing from train.py/play.py into main.py

train()/play() are now presentation-agnostic generators; main.py is
the sole place that turns their EpisodeStep stream into console output,
via new _run_train/_run_play helpers with identical output to before.
EOF
)"
```

---

### Task 6: `PygameRenderer` and the `render` optional dependency

**Files:**

- Modify: `pyproject.toml`
- Create: `renderer.py`

**Interfaces:**

- Consumes: `Board` (`snake_env.py`, Task 1).
- Produces: `PygameRenderer(grid_size: int, cell_size: int = 24, fps: int = 15)` with `.draw(board: Board, episode: int, score: int) -> bool` and `.close() -> None`.

- [ ] **Step 1: Add the `render` optional dependency group**

In `pyproject.toml`, after the existing `dependencies = [...]` block, add:

```toml
[project.optional-dependencies]
render = ["pygame>=2.5"]
```

The full file should read:

```toml
[project]
name = "snake-q-learning"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "pytest>=9.1.1",
    "pytest-cov>=7.1.0",
]

[project.optional-dependencies]
render = ["pygame>=2.5"]

[tool.pytest.ini_options]
pythonpath = ["."]
```

- [ ] **Step 2: Install the extra**

Run: `uv sync --extra render`
Expected: succeeds, installs `pygame` into the project's virtualenv.

- [ ] **Step 3: Implement `PygameRenderer`**

Create `renderer.py`:

```python
import pygame

from snake_env import Board

BG_COLOR = (0, 0, 0)
SNAKE_COLOR = (0, 200, 0)
FOOD_COLOR = (200, 0, 0)


class PygameRenderer:
    def __init__(self, grid_size: int, cell_size: int = 24, fps: int = 15):
        pygame.init()
        self.cell_size = cell_size
        self.fps = fps
        self.screen = pygame.display.set_mode((grid_size * cell_size, grid_size * cell_size))
        self.clock = pygame.time.Clock()

    def draw(self, board: Board, episode: int, score: int) -> bool:
        """Draws one frame. Returns False if the window was closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(BG_COLOR)
        for cell in board.snake_body:
            pygame.draw.rect(self.screen, SNAKE_COLOR, self._rect(cell))
        pygame.draw.rect(self.screen, FOOD_COLOR, self._rect(board.food))
        pygame.display.set_caption(f"episode {episode}  score {score}")
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def _rect(self, cell: tuple[int, int]) -> pygame.Rect:
        x, y = cell
        return pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)

    def close(self) -> None:
        pygame.quit()
```

No pytest coverage is added for this file — `PygameRenderer.__init__` calls `pygame.display.set_mode(...)`, which needs a real (or dummy-driver) display, and this project's testing conventions don't stand up display fixtures. Verified instead by the import-only check below and by manual verification in Task 7.

- [ ] **Step 4: Verify the module imports cleanly**

Run: `uv run --extra render python -c "from renderer import PygameRenderer; print('ok')"`
Expected: prints `ok` with no errors. (This only exercises the import — `pygame.init()`/`set_mode()` aren't called until `PygameRenderer(...)` is constructed.)

- [ ] **Step 5: Run the full test suite to confirm nothing headless broke**

Run: `uv run pytest`
Expected: PASS — unaffected by this task (no test imports `renderer.py`).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml renderer.py
git commit -m "$(cat <<'EOF'
Add PygameRenderer and pygame as an optional 'render' extra

pygame is opt-in (uv sync --extra render) so headless training/play
and their tests never need it importable. PygameRenderer only knows
about pixels and Board snapshots, not episodes or agents.
EOF
)"
```

---

### Task 7: `watch.py`

**Files:**

- Create: `watch.py`

**Interfaces:**

- Consumes: `train()`, `play()` (Tasks 3–4), `PygameRenderer` (Task 6).
- Produces: `watch_train(n_episodes=30_000, grid_size=20, save_path=Path("q_table.json"), render_every=1, cell_size=24, fps=15) -> None`; `watch_play(n_episodes=100, grid_size=20, q_table_path=Path("q_table.json"), cell_size=24, fps=15) -> None`; a `watch.py train|play ...` CLI.

- [ ] **Step 1: Implement `watch.py`**

Create `watch.py`:

```python
import argparse
from pathlib import Path

from play import play
from renderer import PygameRenderer
from train import train


def watch_train(
    n_episodes: int = 30_000,
    grid_size: int = 20,
    save_path: Path = Path("q_table.json"),
    render_every: int = 1,
    cell_size: int = 24,
    fps: int = 15,
) -> None:
    renderer = PygameRenderer(grid_size=grid_size, cell_size=cell_size, fps=fps)
    try:
        for step in train(n_episodes=n_episodes, grid_size=grid_size, save_path=save_path):
            if step.episode % render_every != 0:
                continue
            if not renderer.draw(step.result.board, step.episode, step.result.info["score"]):
                break
    finally:
        renderer.close()


def watch_play(
    n_episodes: int = 100,
    grid_size: int = 20,
    q_table_path: Path = Path("q_table.json"),
    cell_size: int = 24,
    fps: int = 15,
) -> None:
    renderer = PygameRenderer(grid_size=grid_size, cell_size=cell_size, fps=fps)
    try:
        for step in play(n_episodes=n_episodes, grid_size=grid_size, q_table_path=q_table_path):
            if not renderer.draw(step.result.board, step.episode, step.result.info["score"]):
                break
    finally:
        renderer.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="watch.py")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--n-episodes", type=int, default=30_000)
    train_parser.add_argument("--grid-size", type=int, default=20)
    train_parser.add_argument("--save-path", type=Path, default=Path("q_table.json"))
    train_parser.add_argument("--render-every", type=int, default=1)
    train_parser.add_argument("--cell-size", type=int, default=24)
    train_parser.add_argument("--fps", type=int, default=15)

    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--n-episodes", type=int, default=100)
    play_parser.add_argument("--grid-size", type=int, default=20)
    play_parser.add_argument("--q-table-path", type=Path, default=Path("q_table.json"))
    play_parser.add_argument("--cell-size", type=int, default=24)
    play_parser.add_argument("--fps", type=int, default=15)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.mode == "train":
        watch_train(
            n_episodes=args.n_episodes,
            grid_size=args.grid_size,
            save_path=args.save_path,
            render_every=args.render_every,
            cell_size=args.cell_size,
            fps=args.fps,
        )
    elif args.mode == "play":
        watch_play(
            n_episodes=args.n_episodes,
            grid_size=args.grid_size,
            q_table_path=args.q_table_path,
            cell_size=args.cell_size,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
```

`main.py` is untouched by this task and never imports `watch.py` or `pygame`.

- [ ] **Step 2: Verify the CLI parses without opening a window**

Run: `uv run --extra render python watch.py --help`
Expected: prints usage help for the `train`/`play` subcommands, no window opens (argparse's `--help` exits before `watch_train`/`watch_play` is called).

Run: `uv run --extra render python watch.py train --help` and `uv run --extra render python watch.py play --help`
Expected: prints subcommand-specific usage (`--n-episodes`, `--grid-size`, `--render-every`, etc.), no window opens.

- [ ] **Step 3: Manually verify `watch_play` renders correctly**

Run:

```bash
uv run --extra render python main.py train --n-episodes 200 --grid-size 8 --save-path /tmp/watch_q_table.json
uv run --extra render python watch.py play --grid-size 8 --q-table-path /tmp/watch_q_table.json --n-episodes 5
```

Expected: a window opens showing an 8×8 grid; a green snake moves and a red food cell appears in a plausible cell; the window title updates with `episode N score S`; closing the window stops the loop without an unhandled exception; the terminal command returns to a normal prompt.

- [ ] **Step 4: Manually verify `watch_train` and `--render-every` work**

Run: `uv run --extra render python watch.py train --grid-size 8 --n-episodes 50 --render-every 10 --save-path /tmp/watch_train_q_table.json`
Expected: the window renders roughly 5 frames-per-episode-boundary worth of visible episodes (every 10th), not all 50; behavior otherwise matches Step 3; after the run completes, `/tmp/watch_train_q_table.json` exists (confirming `agent.save(...)` still ran, since this run wasn't quit early).

- [ ] **Step 5: Run the full test suite one more time**

Run: `uv run pytest`
Expected: PASS — `watch.py` has no automated tests and doesn't affect any existing ones.

- [ ] **Step 6: Commit**

```bash
git add watch.py
git commit -m "$(cat <<'EOF'
Add watch.py: render a live training run or a saved agent playing

Pure glue over train()/play()'s existing generators and PygameRenderer
- no stepping logic of its own. render_every lets watch_train sample
training episodes instead of rendering all 30,000 of them.
EOF
)"
```
