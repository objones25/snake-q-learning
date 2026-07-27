# Distance-Bucketed State Representation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `SnakeState`'s boolean danger flags and 3-way food-direction signs with distance-bucketed versions, to break through the training plateau observed with the 72-state encoding (average score converges to ~21-22 by episode ~5000 and never improves, even out to 30,000 episodes).

**Architecture:** Three sequential tasks. First, `SnakeState`/`from_world()` is rewritten so danger flags become a 4-value ray-cast distance bucket (0=adjacent, 1=near, 2=medium, 3=far/safe) and food signs become a 5-value magnitude-aware bucket (`{-2,-1,0,1,2}`, splitting "behind"/"ahead" into near/far) — `N_STATES` grows from 72 to 1600 and the index formula becomes a general mixed-radix encoding. Second, `Sign` (now unused anywhere in the codebase) is deleted as cleanup. Third, `QLearningAgent`'s and `train()`'s defaults are bumped to give the ~22x larger state space enough training budget. Task order matters: `SnakeState` must stop using `Sign` (Task 1) before `Sign` can be safely deleted (Task 2), or the suite breaks mid-plan.

**Tech Stack:** Python 3.13, pytest (existing dependency). No new dependencies.

## Global Constraints

- No new dependencies — stdlib only.
- Flat module layout at repo root (matches existing files — no `src/` layout).
- Danger ray-cast: `MAX_DANGER_SCAN = 6`; `_danger_bucket`: `distance == 0` → 0, `distance <= 2` → 1, `distance <= 5` → 2, else → 3. Same tail-exclusion rule as today (`occupied = snake.pos_set - {snake.tail}`).
- Food bucket: `_food_bucket(component)`: `0` → 0; else magnitude `1` if `abs(component) <= 3` else `2`, signed by `component`'s sign. Result range `{-2, -1, 0, 1, 2}`.
- `N_DANGER_BUCKETS = 4`, `N_FOOD_BUCKETS = 5` (per axis). `SnakeState.N_STATES = N_DANGER_BUCKETS ** 3 * N_FOOD_BUCKETS ** 2` (= 1600).
- Index formula: `danger_component = (dng_straight * N_DANGER_BUCKETS + dng_right) * N_DANGER_BUCKETS + dng_left`; `food_component = (food_fwd + 2) * N_FOOD_BUCKETS + (food_lat + 2)`; `index = danger_component * (N_FOOD_BUCKETS ** 2) + food_component`.
- No backward compatibility with an existing `q_table.json` — the shape changes, old saved tables are simply incompatible (already gitignored, no migration).
- No change to reward structure, starvation timeout, or collision rules.
- Training defaults after this plan: `QLearningAgent.__init__`'s `epsilon_decay_episodes` default `100_000`; `train()`'s `n_episodes` default `200_000`.
- Design spec: `docs/superpowers/specs/2026-07-27-distance-bucketed-state-design.md`.

---

### Task 1: Distance-bucketed `SnakeState`

**Files:**
- Modify: `snake_state.py`
- Modify: `tests/test_snake_state.py`

**Interfaces:**
- Consumes: `Snake` (`.head`, `.tail`, `.direction`, `.pos_set`), `Direction` (`.apply()`, `.vec`, `.turn_right()`, `.turn_left()`, `.dx`, `.dy`) — both unchanged from `snake.py`/`snake_types.py`.
- Produces: `SnakeState` with `dng_straight`/`dng_right`/`dng_left: int` (range `0-3`) and `food_fwd`/`food_lat: int` (range `-2..2`), `SnakeState.N_STATES == 1600`, `SnakeState.index -> int` in `[0, 1600)`, `SnakeState.from_world(snake, food, grid_size) -> SnakeState` (same signature as before). Also exports module-level `N_DANGER_BUCKETS = 4` (Task 2's test file imports this for an exhaustive combinatorial check). `q_agent.py`/`train.py` are unaffected — they only ever consume `SnakeState.N_STATES` as an opaque integer.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `tests/test_snake_state.py` with:

```python
import dataclasses
from collections import deque

import pytest

from snake import Snake
from snake_state import SnakeState, N_DANGER_BUCKETS
from snake_types import Direction


def make_state(dng_straight=0, dng_right=0, dng_left=0, food_fwd=0, food_lat=0):
    return SnakeState(
        dng_straight=dng_straight,
        dng_right=dng_right,
        dng_left=dng_left,
        food_fwd=food_fwd,
        food_lat=food_lat,
    )


class TestIndex:
    def test_min_combination_is_zero(self):
        state = make_state(dng_straight=0, dng_right=0, dng_left=0, food_fwd=-2, food_lat=-2)
        assert state.index == 0

    @pytest.mark.parametrize(
        "dng_straight, dng_right, dng_left, expected_danger_component",
        [
            (0, 0, 0, 0),
            (0, 0, 1, 1),
            (0, 0, 2, 2),
            (0, 0, 3, 3),
            (0, 1, 0, 4),
            (1, 0, 0, 16),
            (3, 3, 3, 63),
        ],
    )
    def test_danger_component_scales_index_by_25(
        self, dng_straight, dng_right, dng_left, expected_danger_component
    ):
        state = make_state(
            dng_straight=dng_straight, dng_right=dng_right, dng_left=dng_left,
            food_fwd=-2, food_lat=-2,
        )
        assert state.index == expected_danger_component * 25

    @pytest.mark.parametrize(
        "food_fwd, food_lat, expected_food_component",
        [
            (-2, -2, 0),
            (-2, -1, 1),
            (-2, 0, 2),
            (-2, 1, 3),
            (-2, 2, 4),
            (-1, -2, 5),
            (0, -2, 10),
            (0, 0, 12),
            (1, 0, 17),
            (2, 2, 24),
        ],
    )
    def test_food_component_offsets_index(self, food_fwd, food_lat, expected_food_component):
        state = make_state(food_fwd=food_fwd, food_lat=food_lat)
        assert state.index == expected_food_component

    def test_index_within_bounds_for_all_combinations(self):
        seen = set()
        for dng_straight in range(N_DANGER_BUCKETS):
            for dng_right in range(N_DANGER_BUCKETS):
                for dng_left in range(N_DANGER_BUCKETS):
                    for food_fwd in range(-2, 3):
                        for food_lat in range(-2, 3):
                            state = make_state(
                                dng_straight=dng_straight,
                                dng_right=dng_right,
                                dng_left=dng_left,
                                food_fwd=food_fwd,
                                food_lat=food_lat,
                            )
                            assert 0 <= state.index < SnakeState.N_STATES
                            seen.add(state.index)
        assert len(seen) == SnakeState.N_STATES

    def test_n_states_matches_total_combinations(self):
        assert SnakeState.N_STATES == 4 ** 3 * 5 ** 2


class TestImmutability:
    def test_is_frozen(self):
        state = make_state()
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.dng_straight = 1


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
    def test_dng_straight_bucket_0_at_wall(self, head, direction):
        snake = Snake(head, direction)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=10)
        assert state.dng_straight == 0

    def test_non_tail_body_segment_is_bucket_0(self):
        # tail=(5,3), middle=(5,4), head=(5,5), direction RIGHT.
        # Turning left (UP) steps onto (5,4), a non-tail body segment.
        snake = make_snake([(5, 3), (5, 4), (5, 5)], Direction.RIGHT)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=10)
        assert state.dng_left == 0  # RIGHT.turn_left() == UP -> (5,4) adjacent

    def test_open_space_is_bucket_3(self):
        # (10, 10) in a 20-wide grid keeps a 10-cell margin on every side,
        # well beyond the 6-step scan — (5, 5) is only 5 steps from the
        # y=0 wall, which would put dng_left in bucket 2, not 3.
        snake = Snake((10, 10), Direction.RIGHT)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=20)
        assert state.dng_straight == 3
        assert state.dng_right == 3
        assert state.dng_left == 3

    @pytest.mark.parametrize(
        "obstacle_offset, expected_bucket",
        [
            (2, 1),  # obstacle 2 steps away -> ray distance=1 -> bucket 1
            (3, 1),  # obstacle 3 steps away -> ray distance=2 -> bucket 1
            (4, 2),  # obstacle 4 steps away -> ray distance=3 -> bucket 2
            (6, 2),  # obstacle 6 steps away -> ray distance=5 -> bucket 2
        ],
    )
    def test_danger_bucket_scales_with_obstacle_distance(self, obstacle_offset, expected_bucket):
        head = (5, 5)
        obstacle = (5 + obstacle_offset, 5)
        tail = (0, 0)  # far away, irrelevant to this ray
        snake = make_snake([tail, obstacle, head], Direction.RIGHT)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=20)
        assert state.dng_straight == expected_bucket

    def test_tail_is_passed_through_when_only_obstacle_in_range(self):
        # tail=(6,5) directly ahead of head=(5,5); nothing else nearby.
        # The tail vacates on this move, so the ray should pass through it
        # and report "no danger within range" (bucket 3), not stop at it.
        snake = make_snake([(6, 5), (5, 5)], Direction.RIGHT)  # tail, head
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=20)
        assert state.dng_straight == 3

    def test_ray_continues_past_tail_to_find_obstacle_beyond(self):
        # tail=(6,5) directly ahead of head, then a non-tail segment further
        # ahead at (8,5). The ray should skip the vacating tail and report
        # the distance to the real obstacle beyond it.
        snake = make_snake([(6, 5), (8, 5), (5, 5)], Direction.RIGHT)  # tail, mid, head
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=20)
        assert state.dng_straight == 1


@pytest.mark.parametrize("direction", list(Direction))
class TestFromWorldFoodBuckets:
    def test_food_near_ahead(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        food = (head[0] + 3 * direction.dx, head[1] + 3 * direction.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (1, 0)

    def test_food_far_ahead(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        food = (head[0] + 4 * direction.dx, head[1] + 4 * direction.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (2, 0)

    def test_food_near_behind(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        behind = direction.turn_right().turn_right()
        food = (head[0] + 3 * behind.dx, head[1] + 3 * behind.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (-1, 0)

    def test_food_far_behind(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        behind = direction.turn_right().turn_right()
        food = (head[0] + 4 * behind.dx, head[1] + 4 * behind.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (-2, 0)

    def test_food_near_right(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        right = direction.turn_right()
        food = (head[0] + 3 * right.dx, head[1] + 3 * right.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (0, 1)

    def test_food_far_right(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        right = direction.turn_right()
        food = (head[0] + 4 * right.dx, head[1] + 4 * right.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (0, 2)

    def test_food_near_left(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        left = direction.turn_left()
        food = (head[0] + 3 * left.dx, head[1] + 3 * left.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (0, -1)

    def test_food_far_left(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        left = direction.turn_left()
        food = (head[0] + 4 * left.dx, head[1] + 4 * left.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (0, -2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_snake_state.py -v`
Expected: FAIL — the current `SnakeState` still has boolean/`Sign` fields and the old index formula, so most assertions above will fail (wrong types, wrong index values, `AttributeError` on `N_DANGER_BUCKETS` import).

- [ ] **Step 3: Implement the distance-bucketed `SnakeState`**

Replace the full contents of `snake_state.py` with:

```python
from dataclasses import dataclass

from snake import Snake
from snake_types import Direction

MAX_DANGER_SCAN = 6
N_DANGER_BUCKETS = 4
N_FOOD_BUCKETS = 5


def _ray_distance(
    head: tuple[int, int],
    direction: Direction,
    grid_size: int,
    occupied: set[tuple[int, int]],
    max_scan: int = MAX_DANGER_SCAN,
) -> int:
    cell = head
    for distance in range(max_scan):
        cell = direction.apply(cell)
        in_bounds = 0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size
        if not in_bounds or cell in occupied:
            return distance
    return max_scan


def _danger_bucket(distance: int) -> int:
    if distance == 0:
        return 0
    if distance <= 2:
        return 1
    if distance <= 5:
        return 2
    return 3


def _food_bucket(component: int) -> int:
    if component == 0:
        return 0
    magnitude = 1 if abs(component) <= 3 else 2
    return magnitude if component > 0 else -magnitude


@dataclass(frozen=True, slots=True)
class SnakeState:
    dng_straight: int  # 0-3, see _danger_bucket
    dng_right: int
    dng_left: int
    food_fwd: int  # -2..2, see _food_bucket
    food_lat: int

    N_STATES = N_DANGER_BUCKETS ** 3 * N_FOOD_BUCKETS ** 2

    @property
    def index(self) -> int:
        danger_component = (
            self.dng_straight * N_DANGER_BUCKETS + self.dng_right
        ) * N_DANGER_BUCKETS + self.dng_left
        food_component = (self.food_fwd + 2) * N_FOOD_BUCKETS + (self.food_lat + 2)
        return danger_component * (N_FOOD_BUCKETS ** 2) + food_component

    @classmethod
    def from_world(cls, snake: Snake, food: tuple[int, int], grid_size: int) -> "SnakeState":
        head = snake.head
        direction = snake.direction
        occupied = snake.pos_set - {snake.tail}

        dng_straight = _danger_bucket(_ray_distance(head, direction, grid_size, occupied))
        dng_right = _danger_bucket(_ray_distance(head, direction.turn_right(), grid_size, occupied))
        dng_left = _danger_bucket(_ray_distance(head, direction.turn_left(), grid_size, occupied))

        food_vec = (food[0] - head[0], food[1] - head[1])
        fwd_axis = direction.vec
        right_axis = direction.turn_right().vec
        food_fwd = _food_bucket(food_vec[0] * fwd_axis[0] + food_vec[1] * fwd_axis[1])
        food_lat = _food_bucket(food_vec[0] * right_axis[0] + food_vec[1] * right_axis[1])

        return cls(
            dng_straight=dng_straight,
            dng_right=dng_right,
            dng_left=dng_left,
            food_fwd=food_fwd,
            food_lat=food_lat,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_snake_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add snake_state.py tests/test_snake_state.py
git commit -m "Replace boolean danger flags and food signs with distance buckets"
```

---

### Task 2: Remove unused `Sign` type

**Files:**
- Modify: `snake_types.py`
- Modify: `tests/test_snake_types.py`

**Interfaces:**
- Produces: `snake_types.py` exporting only `Action` and `Direction` (no `Sign`). Nothing downstream references `Sign` after Task 1, so this is a pure deletion with no consumers to update.

- [ ] **Step 1: Update the tests to drop `Sign` (this will fail only if `Sign` is still referenced elsewhere — it should already be gone from `snake_state.py` after Task 1)**

Replace the full contents of `tests/test_snake_types.py` with:

```python
import pytest

from snake_types import Action, Direction


class TestAction:
    def test_values(self):
        assert Action.STRAIGHT == 0
        assert Action.RIGHT == 1
        assert Action.LEFT == 2

    def test_is_int_enum_with_three_members(self):
        assert len(list(Action)) == 3


class TestDirection:
    @pytest.mark.parametrize(
        "direction, vec",
        [
            (Direction.RIGHT, (1, 0)),
            (Direction.DOWN, (0, 1)),
            (Direction.LEFT, (-1, 0)),
            (Direction.UP, (0, -1)),
        ],
    )
    def test_vec_dx_dy(self, direction, vec):
        assert direction.vec == vec
        assert direction.dx == vec[0]
        assert direction.dy == vec[1]

    @pytest.mark.parametrize(
        "start, expected",
        [
            (Direction.RIGHT, Direction.DOWN),
            (Direction.DOWN, Direction.LEFT),
            (Direction.LEFT, Direction.UP),
            (Direction.UP, Direction.RIGHT),
        ],
    )
    def test_turn_right(self, start, expected):
        assert start.turn_right() == expected

    @pytest.mark.parametrize(
        "start, expected",
        [
            (Direction.RIGHT, Direction.UP),
            (Direction.UP, Direction.LEFT),
            (Direction.LEFT, Direction.DOWN),
            (Direction.DOWN, Direction.RIGHT),
        ],
    )
    def test_turn_left(self, start, expected):
        assert start.turn_left() == expected

    def test_turn_right_is_turn_left_inverse(self):
        for direction in Direction:
            assert direction.turn_right().turn_left() == direction
            assert direction.turn_left().turn_right() == direction

    def test_four_right_turns_return_to_start(self):
        for direction in Direction:
            result = direction
            for _ in range(4):
                result = result.turn_right()
            assert result == direction

    @pytest.mark.parametrize(
        "direction, p, expected",
        [
            (Direction.RIGHT, (0, 0), (1, 0)),
            (Direction.DOWN, (0, 0), (0, 1)),
            (Direction.LEFT, (0, 0), (-1, 0)),
            (Direction.UP, (0, 0), (0, -1)),
            (Direction.RIGHT, (3, 4), (4, 4)),
        ],
    )
    def test_apply(self, direction, p, expected):
        assert direction.apply(p) == expected
```

- [ ] **Step 2: Run test to verify it still passes (Sign's removal from this file has no failing case to observe — verify no other file references Sign first)**

Run: `grep -rn "Sign" --include="*.py" . | grep -v .venv`
Expected: only matches inside `snake_types.py` itself (the class definition, to be deleted next). If anything else matches, STOP — Task 1 left a `Sign` reference somewhere it shouldn't have; do not proceed until resolved.

Run: `uv run pytest tests/test_snake_types.py -v`
Expected: PASS (this file no longer imports or references `Sign`)

- [ ] **Step 3: Delete `Sign` from `snake_types.py`**

Replace the full contents of `snake_types.py` with:

```python
from enum import Enum, IntEnum


class Action(IntEnum):
    STRAIGHT = 0
    RIGHT = 1
    LEFT = 2


class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    @property
    def vec(self) -> tuple[int, int]:
        return _VECS[self.value]

    @property
    def dx(self) -> int:
        return _VECS[self.value][0]

    @property
    def dy(self) -> int:
        return _VECS[self.value][1]

    def turn_right(self) -> "Direction":
        return Direction((self.value + 1) % 4)

    def turn_left(self) -> "Direction":
        return Direction((self.value - 1) % 4)

    def apply(self, p: tuple[int, int]) -> tuple[int, int]:
        return (p[0] + self.dx, p[1] + self.dy)


_VECS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))
```

- [ ] **Step 4: Run the full suite to verify nothing broke**

Run: `uv run pytest -v`
Expected: PASS (full suite — `Sign` had no remaining consumers after Task 1)

- [ ] **Step 5: Commit**

```bash
git add snake_types.py tests/test_snake_types.py
git commit -m "Remove unused Sign type"
```

---

### Task 3: Bump training defaults for the larger state space

**Files:**
- Modify: `q_agent.py`
- Modify: `train.py`
- Modify: `tests/test_q_agent.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Produces: `QLearningAgent.__init__`'s `epsilon_decay_episodes` default becomes `100_000` (was `5000`). `train()`'s `n_episodes` default becomes `200_000` (was `10000`). No signature changes, no new parameters — only default values change.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_q_agent.py` (append a new class; keep all existing classes and imports as-is):

```python
class TestDefaults:
    def test_default_epsilon_decay_episodes_is_100000(self):
        agent = QLearningAgent(n_states=5)
        assert agent.epsilon_decay_episodes == 100_000
```

Add to `tests/test_train.py` (append a new class and the `inspect` import; keep all existing classes and imports as-is):

```python
import inspect


class TestDefaults:
    def test_default_n_episodes_is_200000(self):
        assert inspect.signature(train).parameters["n_episodes"].default == 200_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_q_agent.py::TestDefaults tests/test_train.py::TestDefaults -v`
Expected: FAIL — `agent.epsilon_decay_episodes == 5000`, not `100_000`; `train`'s `n_episodes` default is `10000`, not `200_000`.

- [ ] **Step 3: Bump the defaults**

In `q_agent.py`, change the `__init__` signature's `epsilon_decay_episodes` default:

```python
    def __init__(
        self,
        n_states: int,
        n_actions: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_episodes: int = 100_000,
    ):
```

In `train.py`, change the `train()` signature's `n_episodes` default:

```python
def train(
    n_episodes: int = 200_000,
    grid_size: int = 12,
    save_path: str = "q_table.json",
) -> QLearningAgent:
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: PASS (full suite). This does not run a real 200,000-episode training session — all existing `train()`-invoking tests pass `n_episodes` explicitly as a small override, so the new default doesn't slow down the test suite.

- [ ] **Step 5: Commit**

```bash
git add q_agent.py train.py tests/test_q_agent.py tests/test_train.py
git commit -m "Bump training defaults for the larger distance-bucketed state space"
```

---

## Final verification

Run: `uv run pytest -q`
Expected: all tests pass, zero failures.

Optionally, sanity-check the new representation actually helps: `uv run python3 -c "from train import train; agent = train(n_episodes=200000, grid_size=12, save_path='/tmp/q_table_check.json'); print('final epsilon:', agent.epsilon)"` and compare the printed `avg_score` progression against the ~21-22 plateau observed with the old 72-state representation. This is a manual sanity check, not a test — training run duration and outcome quality are not asserted in the automated suite.
