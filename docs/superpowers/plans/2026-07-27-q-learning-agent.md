# Q-Learning Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tabular Q-learning agent that trains against `SnakeEnv`, plus the training loop that runs it — preceded by a small refactor that replaces `SnakeEnv.step()`'s bare tuple return with a named `StepResult` dataclass.

**Architecture:** Three sequential pieces. First, `SnakeEnv.step()`'s return type changes from `tuple[SnakeState, float, bool, dict]` to a `StepResult` dataclass (adds a first-class `truncated` field, replacing the `info["truncated"]` key) — this is a refactor of already-merged code, done before anything depends on the old shape. Second, a new `q_agent.py` holds `QLearningAgent`: a plain-list-backed Q-table, epsilon-greedy action selection with linear decay, the Bellman update (with the truncation-aware bootstrap rule), and JSON save/load. Third, a new `train.py` wires `SnakeEnv` + `QLearningAgent` together in an episode loop with periodic progress logging, and `main.py` becomes the project's entry point by calling it. No rendering, no CLI argument parsing, no neural-network agent — see the design spec for the full non-goals list.

**Tech Stack:** Python 3.13, pytest (existing dependency). No new dependencies — `json`, `random`, `dataclasses`, `collections.deque` are all stdlib.

## Global Constraints

- No new dependencies — stdlib only, plus the project's existing `pytest`/`pytest-cov`.
- Flat module layout at repo root (matches existing `snake.py`, `snake_state.py`, `snake_types.py`, `snake_env.py` — no `src/` layout).
- `pyproject.toml` already sets `pythonpath = ["."]` for pytest.
- `StepResult` is frozen/slotted (`@dataclass(frozen=True, slots=True)`), matching the existing `SnakeState` pattern in `snake_state.py`.
- Q-table: plain `list[list[float]]`, shape `(SnakeState.N_STATES, 3)` — no numpy.
- Reward/hyperparameter defaults (exact values, from the design spec): `alpha=0.1`, `gamma=0.9`, `epsilon_start=1.0`, `epsilon_end=0.01`, `epsilon_decay_episodes=5000`, `n_episodes=10000` (in `train()`'s signature), `save_path="q_table.json"`.
- Bellman target rule: on real death (`done=True, truncated=False`) the target is `reward` alone (no bootstrap); on truncation (`done=True, truncated=True`) and on normal steps (`done=False`) the target is `reward + gamma * max(q_table[next_index])`.
- `choose_action`'s greedy branch ties always resolve to the lowest-index action.
- Design spec: `docs/superpowers/specs/2026-07-27-q-learning-agent-design.md`.

---

### Task 1: `StepResult` dataclass refactor

**Files:**
- Modify: `snake_env.py`
- Modify: `tests/test_snake_env.py`

**Interfaces:**
- Consumes: `SnakeState` (unchanged, from `snake_state.py`).
- Produces: `StepResult(state: SnakeState, reward: float, done: bool, truncated: bool, info: dict)`, a frozen dataclass in `snake_env.py`. `SnakeEnv.step(action: Action) -> StepResult` replaces the old tuple return. Task 3's `train()` calls `env.step(action)` and reads `.state`/`.reward`/`.done`/`.truncated`/`.info["score"]` off the result — these exact attribute names.

- [ ] **Step 1: Update the tests to expect `StepResult` (this will fail against the current tuple-returning `step()`)**

Replace the full contents of `tests/test_snake_env.py` with:

```python
import random
from collections import deque

import pytest

from snake import Snake
from snake_env import SnakeEnv, DEATH_REWARD, FOOD_REWARD, STEP_REWARD
from snake_state import SnakeState
from snake_types import Direction, Action


class TestReset:
    def test_snake_starts_at_grid_center(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        assert env.snake.head == (6, 6)
        assert env.snake.length == 1

    def test_snake_direction_is_a_valid_direction(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        assert env.snake.direction in list(Direction)

    def test_reset_direction_is_randomized_across_resets(self):
        env = SnakeEnv(grid_size=12)
        directions_seen = set()
        for _ in range(50):
            env.reset()
            directions_seen.add(env.snake.direction)
        assert len(directions_seen) > 1

    def test_food_in_bounds(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        x, y = env.food
        assert 0 <= x < 12
        assert 0 <= y < 12

    def test_food_not_on_snake(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        assert env.food not in env.snake.pos_set

    def test_steps_since_food_reset_to_zero(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        assert env.steps_since_food == 0

    def test_reset_returns_a_snake_state(self):
        env = SnakeEnv(grid_size=12)
        state = env.reset()
        assert isinstance(state, SnakeState)


class TestStepTurning:
    def test_straight_keeps_direction(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        original_direction = env.snake.direction
        env.step(Action.STRAIGHT)
        assert env.snake.direction == original_direction

    def test_right_action_turns_right(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        original_direction = env.snake.direction
        env.step(Action.RIGHT)
        assert env.snake.direction == original_direction.turn_right()

    def test_left_action_turns_left(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        original_direction = env.snake.direction
        env.step(Action.LEFT)
        assert env.snake.direction == original_direction.turn_left()


class TestStepCollision:
    def test_wall_collision_ends_episode(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.snake = Snake((11, 5), Direction.RIGHT)
        env.food = (0, 0)
        result = env.step(Action.STRAIGHT)
        assert result.done is True
        assert result.truncated is False
        assert result.reward == DEATH_REWARD
        assert env.snake.head == (11, 5)  # snake not moved on collision

    def test_self_collision_ends_episode(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        snake = Snake((5, 5), Direction.RIGHT)
        snake.body = deque([(3, 3), (6, 5), (5, 5)])  # tail, non-tail, head
        snake.pos_set = set(snake.body)
        env.snake = snake
        env.food = (0, 0)
        result = env.step(Action.STRAIGHT)
        assert result.done is True
        assert result.reward == DEATH_REWARD

    def test_moving_into_vacated_tail_is_not_collision(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        snake = Snake((5, 5), Direction.RIGHT)
        snake.body = deque([(6, 5), (5, 5)])  # tail, head
        snake.pos_set = set(snake.body)
        env.snake = snake
        env.food = (0, 0)
        result = env.step(Action.STRAIGHT)
        assert result.done is False
        assert env.snake.head == (6, 5)
        assert env.snake.length == 2
        assert set(env.snake.body) == env.snake.pos_set


class TestStepFoodAndReward:
    def test_food_consumption_grows_snake_and_gives_reward(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = env.snake.direction.apply(env.snake.head)
        old_length = env.snake.length
        result = env.step(Action.STRAIGHT)
        assert result.reward == FOOD_REWARD
        assert result.done is False
        assert env.snake.length == old_length + 1
        assert env.steps_since_food == 0
        assert env.food not in env.snake.pos_set
        assert result.info["score"] == env.snake.length

    def test_non_eating_move_gives_neutral_reward(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        old_length = env.snake.length
        result = env.step(Action.STRAIGHT)
        assert result.reward == STEP_REWARD
        assert result.done is False
        assert env.snake.length == old_length
        assert env.steps_since_food == 1


class TestStepStarvation:
    def test_not_done_at_boundary(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        env.steps_since_food = 100 * env.snake.length - 1
        result = env.step(Action.STRAIGHT)
        assert result.done is False

    def test_done_once_over_threshold(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        env.steps_since_food = 100 * env.snake.length
        result = env.step(Action.STRAIGHT)
        assert result.done is True
        assert result.reward == STEP_REWARD

    def test_starvation_timeout_marks_result_as_truncated(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        env.steps_since_food = 100 * env.snake.length
        result = env.step(Action.STRAIGHT)
        assert result.done is True
        assert result.truncated is True


class TestStepTruncatedFlag:
    def test_death_path_is_not_truncated(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.snake = Snake((11, 5), Direction.RIGHT)
        env.food = (0, 0)
        result = env.step(Action.STRAIGHT)
        assert result.done is True
        assert result.truncated is False

    def test_food_path_is_not_truncated(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = env.snake.direction.apply(env.snake.head)
        result = env.step(Action.STRAIGHT)
        assert result.reward == FOOD_REWARD
        assert result.truncated is False

    def test_non_eating_move_is_not_truncated(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        result = env.step(Action.STRAIGHT)
        assert result.reward == STEP_REWARD
        assert result.truncated is False


class TestLifecycle:
    def test_random_policy_episodes_hold_invariants_every_step(self):
        # Realistic reset() -> step() -> ... -> done loop under a random
        # policy. This is what would have caught the Snake.move() pos_set
        # corruption bug: single hand-assembled steps never exercised the
        # "head moves into the cell the tail is vacating this same move"
        # case, so the desync went unnoticed by 105 green tests.
        #
        # SnakeEnv itself draws its randomness (start direction, food
        # placement) from the shared `random` module rather than an
        # injected Random instance, so the module-level seed is what we
        # need to pin for a reproducible, deterministic test - a
        # locally-scoped random.Random(...) would only control our own
        # action choices and leave the env's internal randomness (and
        # therefore whether this test actually exercises the bug)
        # dependent on whatever happened to run before it.
        grid_size = 8
        num_episodes = 500
        max_steps_per_episode = 1000
        random.seed(1)

        env = SnakeEnv(grid_size=grid_size)

        for _ in range(num_episodes):
            env.reset()
            reached_done = False

            for _ in range(max_steps_per_episode):
                action = random.choice(list(Action))
                result = env.step(action)

                body = env.snake.body
                assert set(body) == env.snake.pos_set
                assert len(body) == len(set(body))
                for x, y in body:
                    assert 0 <= x < grid_size
                    assert 0 <= y < grid_size

                fx, fy = env.food
                assert 0 <= fx < grid_size
                assert 0 <= fy < grid_size
                assert env.food not in env.snake.pos_set

                assert 0 <= result.state.index < SnakeState.N_STATES
                assert result.reward in (FOOD_REWARD, DEATH_REWARD, STEP_REWARD)
                assert result.info["score"] == env.snake.length

                if result.done:
                    reached_done = True
                    break

            assert reached_done, (
                f"episode did not reach done within {max_steps_per_episode} steps"
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_snake_env.py -v`
Expected: FAIL — errors like `AttributeError: 'tuple' object has no attribute 'done'` (the current `step()` still returns a bare tuple).

- [ ] **Step 3: Implement `StepResult` and update `step()`**

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
class StepResult:
    state: SnakeState
    reward: float
    done: bool
    truncated: bool
    info: dict


class SnakeEnv:
    def __init__(self, grid_size: int = 12):
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
            return StepResult(state, DEATH_REWARD, True, False, {"score": self.snake.length})

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
        return StepResult(state, reward, done, done, {"score": self.snake.length})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: PASS (full suite — this refactor touches only `snake_env.py` and its test file, so `snake.py`/`snake_state.py`/`snake_types.py` tests are unaffected)

- [ ] **Step 5: Commit**

```bash
git add snake_env.py tests/test_snake_env.py
git commit -m "Replace SnakeEnv.step()'s tuple return with a StepResult dataclass"
```

---

### Task 2: `QLearningAgent`

**Files:**
- Create: `q_agent.py`
- Test: `tests/test_q_agent.py`

**Interfaces:**
- Consumes: `Action` (IntEnum, from `snake_types.py`).
- Produces: `QLearningAgent(n_states: int, n_actions: int = 3, alpha: float = 0.1, gamma: float = 0.9, epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay_episodes: int = 5000)` with public attributes `.q_table` (`list[list[float]]`), `.alpha`, `.gamma`, `.epsilon_start`, `.epsilon_end`, `.epsilon_decay_episodes`, `.epsilon`; methods `.set_epsilon_for_episode(episode: int) -> None`, `.choose_action(state_index: int) -> Action`, `.update(state_index: int, action: Action, reward: float, next_index: int, done: bool, truncated: bool) -> None`, `.save(path: str) -> None`, `.load(path: str) -> None`. Task 3's `train()` constructs one with `n_states=SnakeState.N_STATES` and calls all of the above exactly as named.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_q_agent.py`:

```python
import os
import tempfile

import pytest

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
            n_states=5, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_episodes=100
        )
        agent.set_epsilon_for_episode(0)
        assert agent.epsilon == 1.0

    def test_epsilon_reaches_epsilon_end_at_decay_episodes(self):
        agent = QLearningAgent(
            n_states=5, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_episodes=100
        )
        agent.set_epsilon_for_episode(100)
        assert agent.epsilon == pytest.approx(0.01)

    def test_epsilon_holds_at_epsilon_end_past_decay_episodes(self):
        agent = QLearningAgent(
            n_states=5, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_episodes=100
        )
        agent.set_epsilon_for_episode(500)
        assert agent.epsilon == pytest.approx(0.01)

    def test_epsilon_is_linear_at_midpoint(self):
        agent = QLearningAgent(
            n_states=5, epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_episodes=100
        )
        agent.set_epsilon_for_episode(50)
        assert agent.epsilon == pytest.approx(0.5)


class TestUpdate:
    def test_normal_step_bootstraps_with_max_next_q(self):
        agent = QLearningAgent(n_states=5, alpha=0.5, gamma=0.9)
        agent.q_table[1] = [1.0, 2.0, 0.5]  # max = 2.0
        agent.update(
            state_index=0, action=Action.STRAIGHT, reward=1.0,
            next_index=1, done=False, truncated=False,
        )
        # target = 1.0 + 0.9 * 2.0 = 2.8; new_q = 0.0 + 0.5 * (2.8 - 0.0) = 1.4
        assert agent.q_table[0][Action.STRAIGHT] == pytest.approx(1.4)

    def test_real_death_does_not_bootstrap(self):
        agent = QLearningAgent(n_states=5, alpha=0.5, gamma=0.9)
        agent.q_table[1] = [100.0, 100.0, 100.0]  # would blow up the target if bootstrapped
        agent.update(
            state_index=0, action=Action.STRAIGHT, reward=-10.0,
            next_index=1, done=True, truncated=False,
        )
        # target = -10.0 (no bootstrap); new_q = 0.0 + 0.5 * (-10.0 - 0.0) = -5.0
        assert agent.q_table[0][Action.STRAIGHT] == pytest.approx(-5.0)

    def test_truncation_bootstraps_like_a_normal_step(self):
        agent = QLearningAgent(n_states=5, alpha=0.5, gamma=0.9)
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
            path = os.path.join(tmpdir, "q_table.json")
            agent.save(path)

            loaded_agent = QLearningAgent(n_states=3)
            loaded_agent.load(path)
            assert loaded_agent.q_table == agent.q_table
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_q_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'q_agent'`

- [ ] **Step 3: Write the implementation**

Create `q_agent.py`:

```python
import json
import random

from snake_types import Action


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_episodes: int = 5000,
    ):
        self.q_table: list[list[float]] = [[0.0] * n_actions for _ in range(n_states)]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon = epsilon_start

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

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.q_table, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            self.q_table = json.load(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_q_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add q_agent.py tests/test_q_agent.py
git commit -m "Add QLearningAgent"
```

---

### Task 3: Training loop and entry point

**Files:**
- Create: `train.py`
- Modify: `main.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `SnakeEnv`/`StepResult` (`env.reset() -> SnakeState`, `env.step(action) -> StepResult` from Task 1's `snake_env.py`), `QLearningAgent` (from Task 2's `q_agent.py`), `SnakeState.N_STATES` (from `snake_state.py`).
- Produces: `train(n_episodes: int = 10000, grid_size: int = 12, save_path: str = "q_table.json") -> QLearningAgent` in `train.py`. `main.py` calls `train()` with no arguments when run as a script.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train.py`:

```python
import os
import tempfile

from q_agent import QLearningAgent
from snake_state import SnakeState
from train import train


class TestTrain:
    def test_returns_agent_with_correctly_shaped_q_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "q_table.json")
            agent = train(n_episodes=20, grid_size=8, save_path=path)

        assert isinstance(agent, QLearningAgent)
        assert len(agent.q_table) == SnakeState.N_STATES
        assert all(len(row) == 3 for row in agent.q_table)

    def test_epsilon_decreases_from_start_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "q_table.json")
            agent = train(n_episodes=20, grid_size=8, save_path=path)

        assert agent.epsilon < agent.epsilon_start

    def test_saves_q_table_to_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "q_table.json")
            train(n_episodes=20, grid_size=8, save_path=path)
            assert os.path.exists(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train'`

- [ ] **Step 3: Write the implementation**

Create `train.py`:

```python
from collections import deque

from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState


def train(
    n_episodes: int = 10000,
    grid_size: int = 12,
    save_path: str = "q_table.json",
) -> QLearningAgent:
    env = SnakeEnv(grid_size=grid_size)
    agent = QLearningAgent(n_states=SnakeState.N_STATES)
    recent_scores: deque[int] = deque(maxlen=500)

    for episode in range(n_episodes):
        agent.set_epsilon_for_episode(episode)
        state = env.reset()
        result = None
        while result is None or not result.done:
            action = agent.choose_action(state.index)
            result = env.step(action)
            agent.update(
                state.index, action, result.reward,
                result.state.index, result.done, result.truncated,
            )
            state = result.state

        recent_scores.append(result.info["score"])
        if episode % 500 == 0:
            avg_score = sum(recent_scores) / len(recent_scores)
            print(f"episode {episode:6d}  epsilon={agent.epsilon:.3f}  avg_score={avg_score:.2f}")

    agent.save(save_path)
    return agent


if __name__ == "__main__":
    train()
```

Replace the full (currently empty) contents of `main.py` with:

```python
from train import train

if __name__ == "__main__":
    train()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add train.py main.py tests/test_train.py
git commit -m "Add training loop and wire up main.py as the entry point"
```

---

## Final verification

Run: `uv run pytest -q`
Expected: all tests pass (existing `Snake`/`SnakeState`/`SnakeEnv`/`snake_types` tests plus all new tests from Tasks 1-3), zero failures.
