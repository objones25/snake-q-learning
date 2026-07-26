# Snake Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless, Gym-like `SnakeEnv` (reset/step) on top of the existing `Snake` entity, so the Q-learning agent has something to train against.

**Architecture:** Three additions layered on the existing `snake_types.py` / `snake.py` / `snake_state.py` modules: a new `Action` enum (relative turns), a `SnakeState.from_world()` classmethod that derives an observation from raw world state (snake + food + grid bounds), and a new `snake_env.py` module holding `SnakeEnv`, which owns the game loop (movement, collision, reward, termination). No rendering.

**Tech Stack:** Python 3.13, pytest (already a dependency, no new dependencies added).

## Global Constraints

- No new dependencies — implementation uses only the stdlib (`random`, `collections.deque`, `dataclasses`, `enum`) plus the project's existing `pytest`/`pytest-cov`.
- Flat module layout at repo root (matches existing `snake.py`, `snake_state.py`, `snake_types.py` — no `src/` layout).
- No rendering/visualization code of any kind — pygame is a separate, later effort.
- `pyproject.toml` already sets `pythonpath = ["."]` for pytest — new test files under `tests/` can import root modules directly (`from snake_env import SnakeEnv`).
- Reward constants: `FOOD_REWARD = 10`, `DEATH_REWARD = -10`, `STEP_REWARD = 0`.
- Starvation timeout: episode ends (neutral reward) once `steps_since_food > 100 * snake.length`.
- Danger flags exclude the snake's tail cell from self-collision checks (tail vacates on a non-growth move).
- Design spec: `docs/superpowers/specs/2026-07-26-snake-environment-design.md`.

---

### Task 1: `Action` enum

**Files:**

- Modify: `snake_types.py`
- Test: `tests/test_snake_types.py`

**Interfaces:**

- Produces: `Action(IntEnum)` with members `STRAIGHT = 0`, `RIGHT = 1`, `LEFT = 2`, importable as `from snake_types import Action`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_snake_types.py`:

```python
from snake_types import Action, Direction, Sign


class TestAction:
    def test_values(self):
        assert Action.STRAIGHT == 0
        assert Action.RIGHT == 1
        assert Action.LEFT == 2

    def test_is_int_enum_with_three_members(self):
        assert len(list(Action)) == 3
```

(Update the existing `from snake_types import Direction, Sign` import line at the top of the file to include `Action`, as shown above.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_snake_types.py::TestAction -v`
Expected: FAIL with `ImportError: cannot import name 'Action' from 'snake_types'`

- [ ] **Step 3: Write minimal implementation**

In `snake_types.py`, add directly below the `Sign` class (before `class Direction(Enum):`):

```python
class Action(IntEnum):
    STRAIGHT = 0
    RIGHT = 1
    LEFT = 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_snake_types.py -v`
Expected: PASS (all tests in the file, including pre-existing `TestSign`/`TestDirection`)

- [ ] **Step 5: Commit**

```bash
git add snake_types.py tests/test_snake_types.py
git commit -m "Add Action enum for relative turn actions"
```

---

### Task 2: `SnakeState.from_world()`

**Files:**

- Modify: `snake_state.py`
- Test: `tests/test_snake_state.py`

**Interfaces:**

- Consumes: `Snake` (`.head`, `.tail`, `.direction`, `.pos_set` from `snake.py`), `Direction` (`.apply()`, `.vec`, `.turn_right()`, `.turn_left()`), `Sign.of()` (both from `snake_types.py`).
- Produces: `SnakeState.from_world(snake: Snake, food: tuple[int, int], grid_size: int) -> SnakeState`, a classmethod. Later tasks (`SnakeEnv`) call this exact signature to build observations.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_snake_state.py` (append; keep existing `TestIndex`/`TestImmutability` classes):

```python
from collections import deque

from snake import Snake
from snake_state import SnakeState
from snake_types import Direction, Sign


def make_snake(body: list[tuple[int, int]], direction: Direction) -> Snake:
    snake = Snake(body[0], direction)
    snake.body = deque(body)
    snake.pos_set = set(body)
    return snake


class TestFromWorldDanger:
    @pytest.mark.parametrize(
        "head, direction",
        [
            ((9, 5), Direction.RIGHT),
            ((5, 9), Direction.DOWN),
            ((0, 5), Direction.LEFT),
            ((5, 0), Direction.UP),
        ],
    )
    def test_dng_straight_true_at_wall(self, head, direction):
        snake = Snake(head, direction)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=10)
        assert state.dng_straight is True

    def test_no_danger_in_open_space(self):
        snake = Snake((5, 5), Direction.RIGHT)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=10)
        assert state.dng_straight is False
        assert state.dng_right is False
        assert state.dng_left is False

    def test_length_one_snake_has_no_self_danger(self):
        snake = Snake((5, 5), Direction.RIGHT)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=10)
        assert state.dng_straight is False
        assert state.dng_right is False
        assert state.dng_left is False

    def test_tail_cell_excluded_from_danger(self):
        # tail=(5,4), head=(5,5); turning to face UP would step onto the
        # tail cell, which vacates on a non-growth move -> not dangerous.
        snake = make_snake([(5, 4), (5, 5)], Direction.RIGHT)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=10)
        assert state.dng_left is False  # RIGHT.turn_left() == UP

    def test_non_tail_body_segment_is_danger(self):
        # tail=(5,3), middle=(5,4), head=(5,5), direction RIGHT.
        # Turning left (UP) steps onto (5,4), a non-tail body segment.
        snake = make_snake([(5, 3), (5, 4), (5, 5)], Direction.RIGHT)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=10)
        assert state.dng_left is True  # RIGHT.turn_left() == UP -> (5,4)


@pytest.mark.parametrize("direction", list(Direction))
class TestFromWorldFoodSigns:
    def test_food_ahead(self, direction):
        snake = Snake((5, 5), direction)
        food = direction.apply(snake.head)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (Sign.POS, Sign.ZERO)

    def test_food_behind(self, direction):
        snake = Snake((5, 5), direction)
        behind = direction.turn_right().turn_right()
        food = behind.apply(snake.head)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (Sign.NEG, Sign.ZERO)

    def test_food_right(self, direction):
        snake = Snake((5, 5), direction)
        food = direction.turn_right().apply(snake.head)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (Sign.ZERO, Sign.POS)

    def test_food_left(self, direction):
        snake = Snake((5, 5), direction)
        food = direction.turn_left().apply(snake.head)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (Sign.ZERO, Sign.NEG)
```

Add `import pytest` at the top of `tests/test_snake_state.py` if not already present (it already is, from the existing `TestImmutability` class).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_snake_state.py -v`
Expected: FAIL with `AttributeError: type object 'SnakeState' has no attribute 'from_world'`

- [ ] **Step 3: Write minimal implementation**

In `snake_state.py`, update the import line and add the classmethod:

```python
from dataclasses import dataclass

from snake import Snake
from snake_types import Direction, Sign


@dataclass(frozen=True, slots=True)
class SnakeState:
    dng_straight: bool
    dng_right: bool
    dng_left: bool
    food_fwd: Sign  # POS = ahead of the head, NEG = behind it
    food_lat: Sign  # POS = to the head's right, NEG = to its left

    N_STATES = 72

    @property
    def index(self) -> int:
        d = (self.dng_straight << 2) | (self.dng_right << 1) | self.dng_left
        return d * 9 + (self.food_fwd + 1) * 3 + (self.food_lat + 1)

    @classmethod
    def from_world(cls, snake: Snake, food: tuple[int, int], grid_size: int) -> "SnakeState":
        head = snake.head
        direction = snake.direction
        occupied = snake.pos_set - {snake.tail}

        def is_danger(d: Direction) -> bool:
            cell = d.apply(head)
            in_bounds = 0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size
            return not in_bounds or cell in occupied

        dng_straight = is_danger(direction)
        dng_right = is_danger(direction.turn_right())
        dng_left = is_danger(direction.turn_left())

        food_vec = (food[0] - head[0], food[1] - head[1])
        fwd_axis = direction.vec
        right_axis = direction.turn_right().vec
        food_fwd = Sign.of(food_vec[0] * fwd_axis[0] + food_vec[1] * fwd_axis[1])
        food_lat = Sign.of(food_vec[0] * right_axis[0] + food_vec[1] * right_axis[1])

        return cls(
            dng_straight=dng_straight,
            dng_right=dng_right,
            dng_left=dng_left,
            food_fwd=food_fwd,
            food_lat=food_lat,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_snake_state.py -v`
Expected: PASS (all tests, including pre-existing `TestIndex`/`TestImmutability`)

- [ ] **Step 5: Commit**

```bash
git add snake_state.py tests/test_snake_state.py
git commit -m "Add SnakeState.from_world() to derive observations from raw game state"
```

---

### Task 3: `SnakeEnv` — `__init__` and `reset()`

**Files:**

- Create: `snake_env.py`
- Test: `tests/test_snake_env.py`

**Interfaces:**

- Consumes: `Snake(start_pos, direction)` from `snake.py`; `Direction`, `Action` from `snake_types.py`; `SnakeState.from_world(snake, food, grid_size)` from `snake_state.py`.
- Produces: `SnakeEnv(grid_size: int = 12)`, `.reset() -> SnakeState`, public attributes `.grid_size`, `.snake`, `.food`, `.steps_since_food`, and a private helper `._place_food() -> tuple[int, int]`. Task 4 calls `self._place_food()` and reads/writes these same attributes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_snake_env.py`:

```python
import pytest

from snake_env import SnakeEnv
from snake_state import SnakeState
from snake_types import Direction


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_snake_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snake_env'`

- [ ] **Step 3: Write minimal implementation**

Create `snake_env.py`:

```python
import random

from snake import Snake
from snake_state import SnakeState
from snake_types import Direction


FOOD_REWARD = 10
DEATH_REWARD = -10
STEP_REWARD = 0


class SnakeEnv:
    def __init__(self, grid_size: int = 12):
        self.grid_size = grid_size
        self.snake: Snake | None = None
        self.food: tuple[int, int] | None = None
        self.steps_since_food = 0

    def reset(self) -> SnakeState:
        start_pos = (self.grid_size // 2, self.grid_size // 2)
        direction = random.choice(list(Direction))
        self.snake = Snake(start_pos, direction)
        self.food = self._place_food()
        self.steps_since_food = 0
        return SnakeState.from_world(self.snake, self.food, self.grid_size)

    def _place_food(self) -> tuple[int, int]:
        all_cells = {
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        }
        free_cells = list(all_cells - self.snake.pos_set)
        return random.choice(free_cells)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_snake_env.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add snake_env.py tests/test_snake_env.py
git commit -m "Add SnakeEnv with reset()"
```

---

### Task 4: `SnakeEnv.step()`

**Files:**

- Modify: `snake_env.py`
- Test: `tests/test_snake_env.py`

**Interfaces:**

- Consumes: everything from Task 3 (`self.snake`, `self.food`, `self.grid_size`, `self.steps_since_food`, `self._place_food()`), plus `Action` from `snake_types.py` and `Snake.turn_right()`/`turn_left()`/`move(food_consumed)` from `snake.py`.
- Produces: `SnakeEnv.step(action: Action) -> tuple[SnakeState, float, bool, dict]`, the reward constants `FOOD_REWARD`/`DEATH_REWARD`/`STEP_REWARD` from Task 3 remain the module's public reward vocabulary (importable by future training-loop code).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_snake_env.py`:

```python
from collections import deque

from snake import Snake
from snake_env import DEATH_REWARD, FOOD_REWARD, STEP_REWARD
from snake_types import Action


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
        state, reward, done, info = env.step(Action.STRAIGHT)
        assert done is True
        assert reward == DEATH_REWARD
        assert env.snake.head == (11, 5)  # snake not moved on collision

    def test_self_collision_ends_episode(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        snake = Snake((5, 5), Direction.RIGHT)
        snake.body = deque([(3, 3), (6, 5), (5, 5)])  # tail, non-tail, head
        snake.pos_set = set(snake.body)
        env.snake = snake
        env.food = (0, 0)
        state, reward, done, info = env.step(Action.STRAIGHT)
        assert done is True
        assert reward == DEATH_REWARD

    def test_moving_into_vacated_tail_is_not_collision(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        snake = Snake((5, 5), Direction.RIGHT)
        snake.body = deque([(6, 5), (5, 5)])  # tail, head
        snake.pos_set = set(snake.body)
        env.snake = snake
        env.food = (0, 0)
        state, reward, done, info = env.step(Action.STRAIGHT)
        assert done is False
        assert env.snake.head == (6, 5)
        assert env.snake.length == 2


class TestStepFoodAndReward:
    def test_food_consumption_grows_snake_and_gives_reward(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = env.snake.direction.apply(env.snake.head)
        old_length = env.snake.length
        state, reward, done, info = env.step(Action.STRAIGHT)
        assert reward == FOOD_REWARD
        assert done is False
        assert env.snake.length == old_length + 1
        assert env.steps_since_food == 0
        assert env.food not in env.snake.pos_set
        assert info["score"] == env.snake.length

    def test_non_eating_move_gives_neutral_reward(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        old_length = env.snake.length
        state, reward, done, info = env.step(Action.STRAIGHT)
        assert reward == STEP_REWARD
        assert done is False
        assert env.snake.length == old_length
        assert env.steps_since_food == 1


class TestStepStarvation:
    def test_not_done_at_boundary(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        env.steps_since_food = 100 * env.snake.length - 1
        state, reward, done, info = env.step(Action.STRAIGHT)
        assert done is False

    def test_done_once_over_threshold(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        env.steps_since_food = 100 * env.snake.length
        state, reward, done, info = env.step(Action.STRAIGHT)
        assert done is True
        assert reward == STEP_REWARD
```

`Direction` is already imported in `tests/test_snake_env.py` from Task 3 — no change needed there; the new imports above (`deque`, `Snake`, the reward constants, and `Action`) are the only additions.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_snake_env.py -v`
Expected: FAIL with `AttributeError: 'SnakeEnv' object has no attribute 'step'`

- [ ] **Step 3: Write minimal implementation**

Add to `snake_env.py` (inside `SnakeEnv`, after `_place_food`):

```python
    def step(self, action: Action) -> tuple[SnakeState, float, bool, dict]:
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
            return state, DEATH_REWARD, True, {"score": self.snake.length}

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
        return state, reward, done, {"score": self.snake.length}
```

Add `from snake_types import Action, Direction` to `snake_env.py`'s existing `from snake_types import Direction` import line (merge into one line).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest -v`
Expected: PASS (full suite, all tasks combined)

- [ ] **Step 5: Commit**

```bash
git add snake_env.py tests/test_snake_env.py
git commit -m "Add SnakeEnv.step() with collision, reward, and starvation logic"
```

---

## Final verification

Run: `uv run pytest -q`
Expected: all tests pass (existing `Snake`/`SnakeState`/`snake_types` tests plus all new tests from Tasks 1-4), zero failures.
