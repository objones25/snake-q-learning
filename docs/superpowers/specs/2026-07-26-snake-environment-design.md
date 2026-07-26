# Snake Environment — Design

## Purpose

Provide a Gym-like training environment for the Q-learning agent, sitting on
top of the existing `Snake` entity and `SnakeState` observation dataclass.
This is the piece that lets the agent actually play episodes: reset a board,
apply an action, get back a state/reward/done signal. Rendering/visualization
(pygame) is explicitly out of scope and will be built as a separate module
later — the environment must stay usable headless.

## Non-goals

- No rendering, no pygame dependency.
- No "board full" win-condition handling (vanishingly rare during training;
  falls through to normal collision/timeout paths if it ever happens).
- No defensive check against calling `step()` after `done=True` without an
  intervening `reset()` — that's a training-loop contract, not a boundary to
  validate.

## Components

### `snake_types.py` — add `Action`

```python
class Action(IntEnum):
    STRAIGHT = 0
    RIGHT = 1
    LEFT = 2
```

Relative to the snake's current heading (matches `Snake.turn_right()` /
`Snake.turn_left()` and the `dng_straight/right/left` fields already in
`SnakeState` — no wasted action for reversing into the snake's own body).

### `snake_state.py` — add `SnakeState.from_world()`

```python
@classmethod
def from_world(cls, snake: Snake, food: tuple[int, int], grid_size: int) -> "SnakeState":
    ...
```

Pure function of world state → `SnakeState`. Colocated with the dataclass it
produces so it's independently unit-testable, separate from env stepping
logic.

**Danger flags** (`dng_straight`, `dng_right`, `dng_left`): for each of the
three relative directions (current heading, `turn_right()`, `turn_left()`),
the candidate cell is dangerous if it's out of grid bounds, or a member of
`snake.pos_set - {snake.tail}`. The tail is excluded because it vacates on a
non-growth move — the danger signal reflects normal movement, not the exact
growth-aware collision outcome.

**Food signs** (`food_fwd`, `food_lat`): project the vector from head to food
onto the forward axis (`direction.vec`) and right-lateral axis
(`direction.turn_right().vec`) via dot product, then `Sign.of()` each
component.

### `snake_env.py` — new `SnakeEnv`

Gym-like interface:

```python
class SnakeEnv:
    def __init__(self, grid_size: int = 12): ...
    def reset(self) -> SnakeState: ...
    def step(self, action: Action) -> tuple[SnakeState, float, bool, dict]: ...
```

Reward constants (module-level): `FOOD_REWARD = 10`, `DEATH_REWARD = -10`,
`STEP_REWARD = 0`.

#### `reset()`

1. New `Snake` at grid center `(grid_size // 2, grid_size // 2)`, facing a
   randomly chosen `Direction`.
2. Food placed on a uniformly random empty cell: `random.choice` over all
   grid cells minus `snake.pos_set`.
3. `steps_since_food = 0`.
4. Return `SnakeState.from_world(snake, food, grid_size)`.

#### `step(action)`

1. Apply `action` to the snake's direction: `turn_right()` / `turn_left()` /
   no-op for `STRAIGHT`. This happens *before* computing the prospective new
   head, so the action affects this move.
2. Compute prospective `new_head = snake.direction.apply(snake.head)` and
   `food_consumed = new_head == food`.
3. Collision check: `new_head` is out of grid bounds, or in `snake.pos_set`
   (excluding `tail` only when `not food_consumed`, since eating means the
   tail doesn't vacate this move).
4. **If collision:** `done = True`, `reward = DEATH_REWARD`. The snake is
   *not* moved — the returned state reflects the pre-collision world, which
   is fine since Q-learning doesn't bootstrap off the next-state on a
   terminal transition.
5. **Else:** call `snake.move(food_consumed)`.
   - If food eaten: place a new food cell (same random-empty-cell logic as
     `reset()`), `reward = FOOD_REWARD`, `steps_since_food = 0`.
   - Else: `reward = STEP_REWARD`, `steps_since_food += 1`.
6. **Starvation timeout:** if `steps_since_food > 100 * snake.length`,
   `done = True`, `reward = STEP_REWARD` (neutral — this is an efficiency
   cutoff, not a failure, so it doesn't carry the death penalty).
7. Return `(SnakeState.from_world(snake, food, grid_size), reward, done,
   info)` with `info = {"score": snake.length}`.

## Data flow

```text
SnakeEnv.reset() ──> Snake + food + grid_size ──> SnakeState.from_world() ──> SnakeState
                                                                                   │
training loop picks action ◄──────────────────────────────────────────────────────┘
        │
        ▼
SnakeEnv.step(action) ──> turn snake ──> compute prospective head ──> collision?
        │                                                                │
        │                                                          yes  │  no
        │                                                                ▼
        │                                                    snake.move(food_consumed)
        │                                                    food eaten? place new food
        ▼
(SnakeState, reward, done, info)
```

## Error handling

None added beyond what the language gives for free. No validation for
out-of-range `grid_size`, no guard against `step()` before `reset()` — these
are programmer-error conditions in an internal training loop, not external
input.

## Testing

- `tests/test_snake_state.py` (extend): `from_world()` — bounds danger on
  all four grid edges, self-collision danger with tail-exclusion (including
  length-1 snake, where tail == head), food sign combinations (fwd/behind ×
  left/right) for each of the 4 headings.
- `tests/test_snake_env.py` (new):
  - `reset()`: snake in bounds, food not on snake, food in bounds.
  - `step()` with `STRAIGHT`/`RIGHT`/`LEFT` changes direction correctly.
  - Wall collision ends episode with `DEATH_REWARD`, snake unmoved.
  - Self-collision (contrived body) ends episode with `DEATH_REWARD`.
  - Food consumption: reward, growth by 1, new food placed off-snake,
    `steps_since_food` resets.
  - Non-eating move: reward 0, length unchanged, `steps_since_food`
    increments.
  - Starvation timeout: episode ends with neutral reward after
    `100 * length` food-less steps.
