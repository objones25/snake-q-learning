# Centralized Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace scattered scalar parameters across `main.py`/`watch.py`/`train.py`/`play.py`/`q_agent.py`/`renderer.py` with per-layer, frozen `Config` dataclasses living in a new `config.py` module, so every default value has exactly one source of truth.

**Architecture:** A new bottom-layer module `config.py` (peer to `snake_types.py`, zero dependencies on the rest of the codebase) defines `AgentConfig`, `RenderConfig`, `TrainConfig`, `PlayConfig`. `QLearningAgent` and `PygameRenderer` take their respective config object directly at construction; `train()`/`play()`/`watch_train()`/`watch_play()` take config objects instead of scalar kwargs; `main.py`/`watch.py` build those config objects from parsed CLI args, with argparse defaults read off the dataclasses instead of re-typed literals.

**Tech Stack:** Python 3.13, `dataclasses` (stdlib), `pytest`, `argparse`, `pygame` (optional `render` extra).

## Global Constraints

- Every default value must stay numerically identical to today — this is a pure structural refactor, no behavior change. Exact values to preserve: `AgentConfig(n_actions=3, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_episodes=5_000)`, `RenderConfig(cell_size=24, fps=15)`, `TrainConfig(n_episodes=30_000, grid_size=20, save_path=Path("q_table.json"))`, `PlayConfig(n_episodes=100, grid_size=20, q_table_path=Path("q_table.json"))`.
- `AgentConfig.n_actions` gets **no CLI flag** anywhere (tied 1:1 to the 3-member `Action` enum — see `docs/superpowers/specs/2026-07-29-centralized-config-design.md`).
- `config.py` has zero imports from the rest of the codebase — it is a bottom-layer module.
- All config dataclasses are `frozen=True`.
- No `WatchTrainConfig`/`WatchPlayConfig` wrapper types — `watch_train`/`watch_play` take their constituent configs (`TrainConfig`/`PlayConfig`, `RenderConfig`, and a plain `render_every: int` where relevant) as separate parameters.
- `PlayConfig` carries no `AgentConfig` field — `play()` constructs its agent with a bare default `AgentConfig()` internally, since it never learns.
- Tasks are sequenced bottom-up (config → agent/renderer → train/play → main/watch). Each task's test-run step is scoped to that task's own test file, not the full suite — `main.py`/`watch.py` are expected to be transiently broken between Task 2 and Task 6/7 (their old call sites won't match the new `train()`/`play()` signatures until those tasks land). The final task runs the complete suite as the overall sanity check.

---

### Task 1: `config.py` module

**Files:**
- Create: `config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `AgentConfig(n_actions=3, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_episodes=5_000)`, `RenderConfig(cell_size=24, fps=15)`, `TrainConfig(n_episodes=30_000, grid_size=20, save_path=Path("q_table.json"), agent=AgentConfig())`, `PlayConfig(n_episodes=100, grid_size=20, q_table_path=Path("q_table.json"))` — all frozen dataclasses, all importable from `config`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import dataclasses
from pathlib import Path

import pytest

from config import AgentConfig, PlayConfig, RenderConfig, TrainConfig


class TestAgentConfigDefaults:
    def test_defaults_match_current_values(self):
        config = AgentConfig()
        assert config.n_actions == 3
        assert config.alpha == 0.1
        assert config.gamma == 0.9
        assert config.epsilon_start == 1.0
        assert config.epsilon_end == 0.01
        assert config.epsilon_decay_episodes == 5_000

    def test_is_frozen(self):
        config = AgentConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.alpha = 0.5


class TestRenderConfigDefaults:
    def test_defaults_match_current_values(self):
        config = RenderConfig()
        assert config.cell_size == 24
        assert config.fps == 15

    def test_is_frozen(self):
        config = RenderConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.fps = 30


class TestTrainConfigDefaults:
    def test_defaults_match_current_values(self):
        config = TrainConfig()
        assert config.n_episodes == 30_000
        assert config.grid_size == 20
        assert config.save_path == Path("q_table.json")
        assert config.agent == AgentConfig()

    def test_is_frozen(self):
        config = TrainConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.n_episodes = 1


class TestPlayConfigDefaults:
    def test_defaults_match_current_values(self):
        config = PlayConfig()
        assert config.n_episodes == 100
        assert config.grid_size == 20
        assert config.q_table_path == Path("q_table.json")

    def test_is_frozen(self):
        config = PlayConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.n_episodes = 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Write the implementation**

Create `config.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    n_actions: int = 3
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 5_000


@dataclass(frozen=True)
class RenderConfig:
    cell_size: int = 24
    fps: int = 15


@dataclass(frozen=True)
class TrainConfig:
    n_episodes: int = 30_000
    grid_size: int = 20
    save_path: Path = Path("q_table.json")
    agent: AgentConfig = field(default_factory=AgentConfig)


@dataclass(frozen=True)
class PlayConfig:
    n_episodes: int = 100
    grid_size: int = 20
    q_table_path: Path = Path("q_table.json")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "Add config.py with AgentConfig/RenderConfig/TrainConfig/PlayConfig"
```

---

### Task 2: `QLearningAgent` takes `AgentConfig`

**Files:**
- Modify: `q_agent.py`
- Test: `tests/test_q_agent.py` (full rewrite)

**Interfaces:**
- Consumes: `AgentConfig` from Task 1.
- Produces: `QLearningAgent(n_states: int, config: AgentConfig = AgentConfig())` — replaces the old `QLearningAgent(n_states, n_actions=3, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_episodes=5_000)`. All other methods (`set_epsilon_for_episode`, `choose_action`, `update`, `save`, `load`) are unchanged.

- [ ] **Step 1: Write the failing test**

Replace the full contents of `tests/test_q_agent.py`:

```python
import inspect
import tempfile
from pathlib import Path

import pytest

from config import AgentConfig
from q_agent import QLearningAgent
from snake_types import Action


class TestChooseAction:
    def test_epsilon_zero_always_picks_greedy_action(self):
        agent = QLearningAgent(n_states=5)
        agent.epsilon = 0.0
        agent.q_table[2] = [0.1, 0.9, 0.3]
        assert agent.choose_action(2) == Action.RIGHT

    def test_epsilon_one_always_explores(self):
        agent = QLearningAgent(n_states=5)
        agent.epsilon = 1.0
        agent.q_table[0] = [100.0, 0.0, 0.0]  # STRAIGHT is clearly best...
        seen = {agent.choose_action(0) for _ in range(200)}
        # ...but epsilon=1 should still pick every action at least once
        assert seen == {Action.STRAIGHT, Action.RIGHT, Action.LEFT}

    def test_tie_breaks_to_lowest_index_action(self):
        agent = QLearningAgent(n_states=5)
        agent.epsilon = 0.0
        assert agent.q_table[0] == [0.0, 0.0, 0.0]
        assert agent.choose_action(0) == Action.STRAIGHT


class TestSetEpsilonForEpisode:
    def test_epsilon_starts_at_epsilon_start(self):
        agent = QLearningAgent(
            n_states=5,
            config=AgentConfig(epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_episodes=100),
        )
        agent.set_epsilon_for_episode(0)
        assert agent.epsilon == 1.0

    def test_epsilon_reaches_epsilon_end_at_decay_episodes(self):
        agent = QLearningAgent(
            n_states=5,
            config=AgentConfig(epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_episodes=100),
        )
        agent.set_epsilon_for_episode(100)
        assert agent.epsilon == pytest.approx(0.01)

    def test_epsilon_holds_at_epsilon_end_past_decay_episodes(self):
        agent = QLearningAgent(
            n_states=5,
            config=AgentConfig(epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_episodes=100),
        )
        agent.set_epsilon_for_episode(500)
        assert agent.epsilon == pytest.approx(0.01)

    def test_epsilon_is_linear_at_midpoint(self):
        agent = QLearningAgent(
            n_states=5,
            config=AgentConfig(epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_episodes=100),
        )
        agent.set_epsilon_for_episode(50)
        assert agent.epsilon == pytest.approx(0.5)


class TestUpdate:
    def test_normal_step_bootstraps_with_max_next_q(self):
        agent = QLearningAgent(n_states=5, config=AgentConfig(alpha=0.5, gamma=0.9))
        agent.q_table[1] = [1.0, 2.0, 0.5]  # max = 2.0
        agent.update(
            state_index=0, action=Action.STRAIGHT, reward=1.0,
            next_index=1, done=False, truncated=False,
        )
        # target = 1.0 + 0.9 * 2.0 = 2.8; new_q = 0.0 + 0.5 * (2.8 - 0.0) = 1.4
        assert agent.q_table[0][Action.STRAIGHT] == pytest.approx(1.4)

    def test_real_death_does_not_bootstrap(self):
        agent = QLearningAgent(n_states=5, config=AgentConfig(alpha=0.5, gamma=0.9))
        agent.q_table[1] = [100.0, 100.0, 100.0]  # would blow up the target if bootstrapped
        agent.update(
            state_index=0, action=Action.STRAIGHT, reward=-10.0,
            next_index=1, done=True, truncated=False,
        )
        # target = -10.0 (no bootstrap); new_q = 0.0 + 0.5 * (-10.0 - 0.0) = -5.0
        assert agent.q_table[0][Action.STRAIGHT] == pytest.approx(-5.0)

    def test_truncation_bootstraps_like_a_normal_step(self):
        agent = QLearningAgent(n_states=5, config=AgentConfig(alpha=0.5, gamma=0.9))
        agent.q_table[1] = [1.0, 2.0, 0.5]  # max = 2.0
        agent.update(
            state_index=0, action=Action.STRAIGHT, reward=0.0,
            next_index=1, done=True, truncated=True,
        )
        # target = 0.0 + 0.9 * 2.0 = 1.8; new_q = 0.0 + 0.5 * (1.8 - 0.0) = 0.9
        assert agent.q_table[0][Action.STRAIGHT] == pytest.approx(0.9)


class TestSaveLoad:
    def test_round_trips_q_table_exactly(self):
        agent = QLearningAgent(n_states=3)
        agent.q_table[0] = [1.0, 2.0, 3.0]
        agent.q_table[1] = [4.0, 5.0, 6.0]
        agent.q_table[2] = [7.0, 8.0, 9.0]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q_table.json"
            agent.save(path)

            loaded_agent = QLearningAgent(n_states=3)
            loaded_agent.load(path)
            assert loaded_agent.q_table == agent.q_table

    def test_save_and_load_path_params_are_path_typed(self):
        assert inspect.signature(QLearningAgent.save).parameters["path"].annotation is Path
        assert inspect.signature(QLearningAgent.load).parameters["path"].annotation is Path


class TestDefaults:
    def test_default_epsilon_decay_episodes_is_5000(self):
        agent = QLearningAgent(n_states=5)
        assert agent.epsilon_decay_episodes == 5_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_q_agent.py -v`
Expected: FAIL — `TypeError: QLearningAgent.__init__() got an unexpected keyword argument 'config'`

- [ ] **Step 3: Write the implementation**

Replace the full contents of `q_agent.py`:

```python
import json
import random
from pathlib import Path

from config import AgentConfig
from snake_types import Action


class QLearningAgent:
    def __init__(self, n_states: int, config: AgentConfig = AgentConfig()):
        self.q_table: list[list[float]] = [[0.0] * config.n_actions for _ in range(n_states)]
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay_episodes = config.epsilon_decay_episodes
        self.epsilon = config.epsilon_start

    def set_epsilon_for_episode(self, episode: int) -> None:
        fraction = min(episode / self.epsilon_decay_episodes, 1.0)
        self.epsilon = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_end)

    def choose_action(self, state_index: int) -> Action:
        if random.random() < self.epsilon:
            return random.choice(list(Action))
        q_values = self.q_table[state_index]
        best_action = max(range(len(q_values)), key=lambda a: q_values[a])
        return Action(best_action)

    def update(
        self,
        state_index: int,
        action: Action,
        reward: float,
        next_index: int,
        done: bool,
        truncated: bool,
    ) -> None:
        current = self.q_table[state_index][action]
        if done and not truncated:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_index])
        self.q_table[state_index][action] += self.alpha * (target - current)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.q_table, f)

    def load(self, path: Path) -> None:
        with open(path) as f:
            self.q_table = json.load(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_q_agent.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add q_agent.py tests/test_q_agent.py
git commit -m "QLearningAgent takes AgentConfig instead of scalar hyperparameters"
```

---

### Task 3: `PygameRenderer` takes `RenderConfig`

**Files:**
- Modify: `renderer.py`

**Interfaces:**
- Consumes: `RenderConfig` from Task 1.
- Produces: `PygameRenderer(grid_size: int, config: RenderConfig = RenderConfig())` — replaces the old `PygameRenderer(grid_size, cell_size=24, fps=15)`. `draw`, `_rect`, `close` are unchanged.

There is no existing `tests/test_renderer.py` — instantiating `PygameRenderer` calls `pygame.init()` and opens a display, and `pygame` is only installed via the optional `render` extra, so this module has never had pytest coverage (consistent with the rest of the test suite, which never imports `renderer.py`). This task is verified with a manual smoke script instead, using SDL's dummy video driver so it doesn't need a real display.

- [ ] **Step 1: Write the implementation**

Replace the full contents of `renderer.py`:

```python
import pygame

from config import RenderConfig
from snake_env import Board

BG_COLOR = (0, 0, 0)
SNAKE_COLOR = (0, 200, 0)
FOOD_COLOR = (200, 0, 0)


class PygameRenderer:
    def __init__(self, grid_size: int, config: RenderConfig = RenderConfig()):
        pygame.init()
        self.cell_size = config.cell_size
        self.fps = config.fps
        self.screen = pygame.display.set_mode(
            (grid_size * self.cell_size, grid_size * self.cell_size)
        )
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

- [ ] **Step 2: Manual verification**

Run:

```bash
SDL_VIDEODRIVER=dummy uv run --extra render python -c "
from config import RenderConfig
from renderer import PygameRenderer

r = PygameRenderer(grid_size=8)
assert r.cell_size == 24
assert r.fps == 15
r.close()

r2 = PygameRenderer(grid_size=8, config=RenderConfig(cell_size=10, fps=5))
assert r2.cell_size == 10
assert r2.fps == 5
r2.close()
print('renderer smoke test passed')
"
```

Expected output: `renderer smoke test passed`

- [ ] **Step 3: Commit**

```bash
git add renderer.py
git commit -m "PygameRenderer takes RenderConfig instead of scalar cell_size/fps"
```

---

### Task 4: `train()` takes `TrainConfig`

**Files:**
- Modify: `train.py`
- Test: `tests/test_train.py` (full rewrite)

**Interfaces:**
- Consumes: `TrainConfig`, `AgentConfig` from Task 1; `QLearningAgent(n_states, config: AgentConfig)` from Task 2.
- Produces: `train(config: TrainConfig = TrainConfig()) -> Iterator[EpisodeStep]` — replaces `train(n_episodes=30_000, grid_size=20, save_path=Path("q_table.json"))`.

- [ ] **Step 1: Write the failing test**

Replace the full contents of `tests/test_train.py`:

```python
import tempfile
from pathlib import Path

import pytest

from config import AgentConfig, TrainConfig
from q_agent import QLearningAgent
from snake_state import SnakeState
from train import train


class TestTrain:
    def test_returns_agent_with_correctly_shaped_q_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q_table.json"
            config = TrainConfig(n_episodes=20, grid_size=8, save_path=path)
            steps = list(train(config))
            agent = steps[-1].agent

        assert isinstance(agent, QLearningAgent)
        assert len(agent.q_table) == SnakeState.N_STATES
        assert all(len(row) == 3 for row in agent.q_table)

    def test_epsilon_decreases_from_start_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q_table.json"
            config = TrainConfig(n_episodes=20, grid_size=8, save_path=path)
            steps = list(train(config))
            agent = steps[-1].agent

        assert agent.epsilon == pytest.approx(0.996238)

    def test_saves_q_table_to_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q_table.json"
            list(train(TrainConfig(n_episodes=20, grid_size=8, save_path=path)))
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
            config = TrainConfig(n_episodes=20, grid_size=8, save_path=Path(tmpdir) / "q.json")
            list(train(config))

        assert (True, False) in seen


class TestDefaults:
    def test_default_n_episodes_is_30000(self):
        assert TrainConfig().n_episodes == 30_000

    def test_default_grid_size_is_20(self):
        assert TrainConfig().grid_size == 20

    def test_save_path_is_path_typed_with_q_table_json_default(self):
        config = TrainConfig()
        assert isinstance(config.save_path, Path)
        assert config.save_path == Path("q_table.json")

    def test_default_agent_config_matches_agent_config_defaults(self):
        assert TrainConfig().agent == AgentConfig()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL — `TypeError: train() takes from 0 to 1 positional arguments but ...` or similar (old `train()` still takes `n_episodes`/`grid_size`/`save_path` kwargs, not a `config` positional)

- [ ] **Step 3: Write the implementation**

Replace the full contents of `train.py`:

```python
from collections.abc import Iterator

from config import TrainConfig
from episode_step import EpisodeStep
from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState


def train(config: TrainConfig = TrainConfig()) -> Iterator[EpisodeStep]:
    env = SnakeEnv(grid_size=config.grid_size)
    agent = QLearningAgent(n_states=SnakeState.N_STATES, config=config.agent)

    for episode in range(config.n_episodes):
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

    agent.save(config.save_path)


if __name__ == "__main__":
    for _ in train():
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "train() takes TrainConfig instead of scalar kwargs"
```

Note: `main.py` and `watch.py` still call the old `train(n_episodes=..., grid_size=..., save_path=...)` signature at this point and will raise `TypeError` if run — this is expected and resolved by Task 6/7.

---

### Task 5: `play()` takes `PlayConfig`

**Files:**
- Modify: `play.py`
- Test: `tests/test_play.py` (full rewrite)

**Interfaces:**
- Consumes: `PlayConfig`, `AgentConfig` from Task 1; `QLearningAgent(n_states, config: AgentConfig)` from Task 2.
- Produces: `play(config: PlayConfig = PlayConfig()) -> Iterator[EpisodeStep]` — replaces `play(n_episodes=100, grid_size=20, q_table_path=Path("q_table.json"))`.

- [ ] **Step 1: Write the failing test**

Replace the full contents of `tests/test_play.py`:

```python
import tempfile
from pathlib import Path

import pytest

from config import PlayConfig
from play import play
from q_agent import QLearningAgent
from snake_state import SnakeState


class TestMissingQTable:
    def test_raises_clear_error_when_q_table_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "does_not_exist.json"
            config = PlayConfig(n_episodes=1, grid_size=8, q_table_path=missing_path)
            with pytest.raises(FileNotFoundError, match=str(missing_path)):
                list(play(config))


class TestPlay:
    def _make_q_table(self, tmpdir):
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        path = Path(tmpdir) / "q_table.json"
        agent.save(path)
        return path

    def test_runs_n_episodes_and_returns_a_score_per_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_q_table(tmpdir)
            config = PlayConfig(n_episodes=5, grid_size=8, q_table_path=path)
            steps = list(play(config))

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
            config = PlayConfig(n_episodes=1, grid_size=8, q_table_path=path)
            list(play(config))

        assert seen_epsilons
        assert all(epsilon == 0.0 for epsilon in seen_epsilons)


class TestDefaults:
    def test_default_n_episodes_is_100(self):
        assert PlayConfig().n_episodes == 100

    def test_default_grid_size_is_20(self):
        assert PlayConfig().grid_size == 20

    def test_default_q_table_path_is_path_typed(self):
        config = PlayConfig()
        assert isinstance(config.q_table_path, Path)
        assert config.q_table_path == Path("q_table.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_play.py -v`
Expected: FAIL — `TypeError` (old `play()` still takes `n_episodes`/`grid_size`/`q_table_path` kwargs, not a `config` positional)

- [ ] **Step 3: Write the implementation**

Replace the full contents of `play.py`:

```python
from collections.abc import Iterator

from config import AgentConfig, PlayConfig
from episode_step import EpisodeStep
from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState


def play(config: PlayConfig = PlayConfig()) -> Iterator[EpisodeStep]:
    if not config.q_table_path.exists():
        raise FileNotFoundError(
            f"No q_table found at {config.q_table_path} — run `main.py train` first"
        )

    env = SnakeEnv(grid_size=config.grid_size)
    agent = QLearningAgent(n_states=SnakeState.N_STATES, config=AgentConfig())
    agent.load(config.q_table_path)
    agent.epsilon = 0.0

    for episode in range(config.n_episodes):
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

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_play.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add play.py tests/test_play.py
git commit -m "play() takes PlayConfig instead of scalar kwargs"
```

---

### Task 6: `main.py` CLI wiring

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py` (full rewrite)

**Interfaces:**
- Consumes: `TrainConfig`, `PlayConfig`, `AgentConfig` from Task 1; `train(config: TrainConfig)` from Task 4; `play(config: PlayConfig)` from Task 5.
- Produces: `main(argv)` unchanged externally (same subcommands, same flags, plus new `--alpha`/`--gamma`/`--epsilon-start`/`--epsilon-end`/`--epsilon-decay-episodes` flags on the `train` subcommand).

- [ ] **Step 1: Write the failing test**

Replace the full contents of `tests/test_main.py`:

```python
from pathlib import Path

import pytest

import main
from config import AgentConfig, PlayConfig, TrainConfig
from q_agent import QLearningAgent
from snake_state import SnakeState


class TestTrainDispatch:
    def test_train_subcommand_calls_train_with_defaults(self, monkeypatch):
        seen_configs = []

        def fake_train(config):
            seen_configs.append(config)
            return iter(())

        monkeypatch.setattr(main, "train", fake_train)

        main.main(["train"])

        assert seen_configs == [TrainConfig()]

    def test_train_subcommand_honors_overrides(self, monkeypatch):
        seen_configs = []

        def fake_train(config):
            seen_configs.append(config)
            return iter(())

        monkeypatch.setattr(main, "train", fake_train)

        main.main(
            ["train", "--n-episodes", "500", "--grid-size", "10", "--save-path", "out.json"]
        )

        assert seen_configs == [
            TrainConfig(n_episodes=500, grid_size=10, save_path=Path("out.json"))
        ]

    def test_train_subcommand_honors_agent_hyperparameter_overrides(self, monkeypatch):
        seen_configs = []

        def fake_train(config):
            seen_configs.append(config)
            return iter(())

        monkeypatch.setattr(main, "train", fake_train)

        main.main(
            [
                "train",
                "--alpha", "0.5",
                "--gamma", "0.8",
                "--epsilon-start", "0.9",
                "--epsilon-end", "0.05",
                "--epsilon-decay-episodes", "1000",
            ]
        )

        assert seen_configs == [
            TrainConfig(
                agent=AgentConfig(
                    alpha=0.5,
                    gamma=0.8,
                    epsilon_start=0.9,
                    epsilon_end=0.05,
                    epsilon_decay_episodes=1000,
                )
            )
        ]


class TestPlayDispatch:
    def test_play_subcommand_calls_play_with_defaults(self, monkeypatch):
        seen_configs = []

        def fake_play(config):
            seen_configs.append(config)
            return iter(())

        monkeypatch.setattr(main, "play", fake_play)

        main.main(["play"])

        assert seen_configs == [PlayConfig()]

    def test_play_subcommand_honors_overrides(self, monkeypatch):
        seen_configs = []

        def fake_play(config):
            seen_configs.append(config)
            return iter(())

        monkeypatch.setattr(main, "play", fake_play)

        main.main(
            ["play", "--n-episodes", "5", "--grid-size", "10", "--q-table-path", "other.json"]
        )

        assert seen_configs == [
            PlayConfig(n_episodes=5, grid_size=10, q_table_path=Path("other.json"))
        ]


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

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_main.py -v`
Expected: FAIL — old `main.py` still builds `fake_train`/`fake_play` kwarg calls (`n_episodes=`, `grid_size=`, `save_path=`) and has no `--alpha`/`--gamma`/etc. flags, so `seen_configs == [TrainConfig()]`-style assertions fail and `--alpha` is an unrecognized argument.

- [ ] **Step 3: Write the implementation**

Replace the full contents of `main.py`:

```python
import argparse
from collections import deque
from pathlib import Path

from config import AgentConfig, PlayConfig, TrainConfig
from play import play
from train import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_defaults = TrainConfig()
    agent_defaults = train_defaults.agent
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--n-episodes", type=int, default=train_defaults.n_episodes)
    train_parser.add_argument("--grid-size", type=int, default=train_defaults.grid_size)
    train_parser.add_argument("--save-path", type=Path, default=train_defaults.save_path)
    train_parser.add_argument("--alpha", type=float, default=agent_defaults.alpha)
    train_parser.add_argument("--gamma", type=float, default=agent_defaults.gamma)
    train_parser.add_argument("--epsilon-start", type=float, default=agent_defaults.epsilon_start)
    train_parser.add_argument("--epsilon-end", type=float, default=agent_defaults.epsilon_end)
    train_parser.add_argument(
        "--epsilon-decay-episodes", type=int, default=agent_defaults.epsilon_decay_episodes
    )

    play_defaults = PlayConfig()
    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--n-episodes", type=int, default=play_defaults.n_episodes)
    play_parser.add_argument("--grid-size", type=int, default=play_defaults.grid_size)
    play_parser.add_argument("--q-table-path", type=Path, default=play_defaults.q_table_path)

    return parser


def _run_train(config: TrainConfig) -> None:
    recent_scores: deque[int] = deque(maxlen=500)
    top_score = 0
    for step in train(config):
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


def _run_play(config: PlayConfig) -> None:
    scores = []
    for step in play(config):
        if not step.result.done:
            continue
        score = step.result.info["score"]
        scores.append(score)
        print(f"episode {step.episode:6d}  score={score}")

    if scores:  # nothing ran (e.g. --n-episodes 0); avoid dividing by zero
        avg_score = sum(scores) / len(scores)
        print(f"avg_score={avg_score:.2f}  top_score={max(scores)}")


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.mode == "train":
        agent_config = AgentConfig(
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_episodes=args.epsilon_decay_episodes,
        )
        config = TrainConfig(
            n_episodes=args.n_episodes,
            grid_size=args.grid_size,
            save_path=args.save_path,
            agent=agent_config,
        )
        _run_train(config)
    elif args.mode == "play":
        config = PlayConfig(
            n_episodes=args.n_episodes,
            grid_size=args.grid_size,
            q_table_path=args.q_table_path,
        )
        _run_play(config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_main.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "main.py builds TrainConfig/PlayConfig from CLI args; add agent hyperparameter flags"
```

---

### Task 7: `watch.py` CLI wiring

**Files:**
- Modify: `watch.py`

**Interfaces:**
- Consumes: `TrainConfig`, `PlayConfig`, `AgentConfig`, `RenderConfig` from Task 1; `train(config: TrainConfig)` from Task 4; `play(config: PlayConfig)` from Task 5; `PygameRenderer(grid_size, config: RenderConfig)` from Task 3.
- Produces: `watch_train(train_config: TrainConfig = TrainConfig(), render_config: RenderConfig = RenderConfig(), render_every: int = 1) -> None` and `watch_play(play_config: PlayConfig = PlayConfig(), render_config: RenderConfig = RenderConfig()) -> None` — replace the old all-scalar signatures. Same CLI flags as before, plus the new `--alpha`/`--gamma`/`--epsilon-*` flags on the `train` subcommand (matching `main.py`).

There is no existing `tests/test_watch.py` (same reasoning as Task 3 — `watch.py` imports `renderer.py`, which requires the optional `pygame` extra). This task is verified with a manual end-to-end smoke run instead.

- [ ] **Step 1: Write the implementation**

Replace the full contents of `watch.py`:

```python
import argparse
from pathlib import Path

from config import AgentConfig, PlayConfig, RenderConfig, TrainConfig
from play import play
from renderer import PygameRenderer
from train import train


def watch_train(
    train_config: TrainConfig = TrainConfig(),
    render_config: RenderConfig = RenderConfig(),
    render_every: int = 1,
) -> None:
    renderer = PygameRenderer(grid_size=train_config.grid_size, config=render_config)
    try:
        for step in train(train_config):
            if step.episode % render_every != 0:
                continue
            if not renderer.draw(step.result.board, step.episode, step.result.info["score"]):
                break
    finally:
        renderer.close()


def watch_play(
    play_config: PlayConfig = PlayConfig(),
    render_config: RenderConfig = RenderConfig(),
) -> None:
    renderer = PygameRenderer(grid_size=play_config.grid_size, config=render_config)
    try:
        for step in play(play_config):
            if not renderer.draw(step.result.board, step.episode, step.result.info["score"]):
                break
    finally:
        renderer.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="watch.py")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_defaults = TrainConfig()
    agent_defaults = train_defaults.agent
    render_defaults = RenderConfig()

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--n-episodes", type=int, default=train_defaults.n_episodes)
    train_parser.add_argument("--grid-size", type=int, default=train_defaults.grid_size)
    train_parser.add_argument("--save-path", type=Path, default=train_defaults.save_path)
    train_parser.add_argument("--render-every", type=int, default=1)
    train_parser.add_argument("--cell-size", type=int, default=render_defaults.cell_size)
    train_parser.add_argument("--fps", type=int, default=render_defaults.fps)
    train_parser.add_argument("--alpha", type=float, default=agent_defaults.alpha)
    train_parser.add_argument("--gamma", type=float, default=agent_defaults.gamma)
    train_parser.add_argument("--epsilon-start", type=float, default=agent_defaults.epsilon_start)
    train_parser.add_argument("--epsilon-end", type=float, default=agent_defaults.epsilon_end)
    train_parser.add_argument(
        "--epsilon-decay-episodes", type=int, default=agent_defaults.epsilon_decay_episodes
    )

    play_defaults = PlayConfig()
    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--n-episodes", type=int, default=play_defaults.n_episodes)
    play_parser.add_argument("--grid-size", type=int, default=play_defaults.grid_size)
    play_parser.add_argument("--q-table-path", type=Path, default=play_defaults.q_table_path)
    play_parser.add_argument("--cell-size", type=int, default=render_defaults.cell_size)
    play_parser.add_argument("--fps", type=int, default=render_defaults.fps)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.mode == "train":
        train_config = TrainConfig(
            n_episodes=args.n_episodes,
            grid_size=args.grid_size,
            save_path=args.save_path,
            agent=AgentConfig(
                alpha=args.alpha,
                gamma=args.gamma,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay_episodes=args.epsilon_decay_episodes,
            ),
        )
        render_config = RenderConfig(cell_size=args.cell_size, fps=args.fps)
        watch_train(train_config, render_config, render_every=args.render_every)
    elif args.mode == "play":
        play_config = PlayConfig(
            n_episodes=args.n_episodes,
            grid_size=args.grid_size,
            q_table_path=args.q_table_path,
        )
        render_config = RenderConfig(cell_size=args.cell_size, fps=args.fps)
        watch_play(play_config, render_config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual verification**

Run from the project root (uses the SDL dummy driver so no real display is needed; produces a scratch q_table first, then watches `play`/`train` for a couple of episodes on a small grid):

```bash
mkdir -p /tmp/watch_smoke
SDL_VIDEODRIVER=dummy uv run python main.py train --n-episodes 5 --grid-size 8 --save-path /tmp/watch_smoke/q_table.json
SDL_VIDEODRIVER=dummy uv run --extra render python watch.py play --n-episodes 2 --grid-size 8 --q-table-path /tmp/watch_smoke/q_table.json --fps 60
SDL_VIDEODRIVER=dummy uv run --extra render python watch.py train --n-episodes 5 --grid-size 8 --save-path /tmp/watch_smoke/q_table2.json --render-every 2 --alpha 0.2 --fps 60
```

Expected: all three invocations exit cleanly (return code 0) with no traceback.

- [ ] **Step 3: Commit**

```bash
git add watch.py
git commit -m "watch.py builds TrainConfig/PlayConfig/RenderConfig from CLI args"
```

---

### Task 8: `CLAUDE.md` updates and full-suite verification

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: nothing new — documentation only.

- [ ] **Step 1: Update the `QLearningAgent` bullet**

In `CLAUDE.md`, find this sentence inside the `QLearningAgent` bullet under "Architecture":

```
- **`QLearningAgent`** is deliberately decoupled from everything above — it only needs `SnakeState.N_STATES` at construction and thereafter only sees integer state indices and `Action` values. Q-table is a plain `list[list[float]]`, persisted as unversioned JSON (`save`/`load`, path-typed via `pathlib.Path`) with no shape validation against `n_states`/`n_actions`.
```

Replace it with:

```
- **`QLearningAgent`** is deliberately decoupled from everything above — it only needs `SnakeState.N_STATES` and an `AgentConfig` (`config.py`) at construction and thereafter only sees integer state indices and `Action` values. Q-table is a plain `list[list[float]]`, persisted as unversioned JSON (`save`/`load`, path-typed via `pathlib.Path`) with no shape validation against `n_states`/`n_actions`.
```

- [ ] **Step 2: Add a `config.py` paragraph to the Architecture section**

Immediately after the layering diagram's closing triple-backtick fence and before the "Off to the side, a separate optional branch renders gameplay..." paragraph, insert:

```
A bottom-layer module, `config.py`, sits alongside `snake_types.py` — frozen dataclasses (`AgentConfig`, `RenderConfig`, `TrainConfig`, `PlayConfig`) that any layer imports for its own parameters, replacing what used to be scalar kwargs duplicated across `main.py`/`watch.py`/`train.py`/`play.py`/`q_agent.py`/`renderer.py`. `QLearningAgent` takes an `AgentConfig`, `PygameRenderer` takes a `RenderConfig`, and `train()`/`play()` take a `TrainConfig`/`PlayConfig` (`TrainConfig` itself carries an `AgentConfig`). See `docs/superpowers/specs/2026-07-29-centralized-config-design.md`.
```

- [ ] **Step 3: Update the "Known spec/code discrepancy" paragraph**

Find:

```
**Known spec/code discrepancy**: `docs/superpowers/specs/2026-07-27-distance-bucketed-state-design.md` calls for bumping `QLearningAgent.epsilon_decay_episodes` to 100,000 and `train()`'s `n_episodes` to 200,000 to match the 22x larger state space (1600 vs. 72 states). Current code still defaults to `epsilon_decay_episodes=5_000` (`q_agent.py`) and `n_episodes=30_000` (`train.py`) — these were deliberately reverted (see commit `17a4a3e`, "revert training defaults"), and tests assert the smaller values. Don't "fix" these back to the spec's numbers without checking why they were reverted first.
```

Replace it with:

```
**Known spec/code discrepancy**: `docs/superpowers/specs/2026-07-27-distance-bucketed-state-design.md` calls for bumping `QLearningAgent.epsilon_decay_episodes` to 100,000 and `train()`'s `n_episodes` to 200,000 to match the 22x larger state space (1600 vs. 72 states). Current code still defaults to `AgentConfig.epsilon_decay_episodes=5_000` and `TrainConfig.n_episodes=30_000` (both in `config.py` — see `docs/superpowers/specs/2026-07-29-centralized-config-design.md`) — these were deliberately reverted (see commit `17a4a3e`, "revert training defaults"), and tests assert the smaller values. Don't "fix" these back to the spec's numbers without checking why they were reverted first.
```

- [ ] **Step 4: Add an agent-hyperparameter example to the Commands section**

Immediately after this line in the `Commands` section:

```
uv run python main.py train --n-episodes 5000 --grid-size 10 --save-path other.json
```

add:

```
uv run python main.py train --alpha 0.05 --gamma 0.95 --epsilon-decay-episodes 10000  # override AgentConfig hyperparameters
```

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest`
Expected: PASS, all tests (config, q_agent, train, play, main, plus all the untouched snake/snake_env/snake_state/snake_types/episode_step tests).

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "Update CLAUDE.md for the centralized config.py refactor"
```
