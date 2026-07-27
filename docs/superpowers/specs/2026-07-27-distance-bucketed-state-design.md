# Distance-Bucketed State Representation — Design

## Purpose

Replace the boolean danger flags and 3-way food-direction signs in
`SnakeState` with distance-bucketed versions of the same five features, to
break through the training plateau observed with the current 72-state
representation (average score converges to ~21-22 by episode ~5000-6000 and
never improves further, even out to 30,000 episodes). The current encoding
can't distinguish "obstacle one step away" from "obstacle five steps away,"
nor "food two cells ahead" from "food across the whole board" — both are
real information a Q-learning agent needs to route around a growing body
and navigate toward food efficiently.

## Non-goals

- No lookahead/planning search (simulating k moves ahead). That was the
  other option considered; it's a separate, larger architectural change
  (requires safe environment-forking/simulation machinery, changes
  per-decision cost from O(1) to O(3^k)) and doesn't address the same root
  cause — it would still query the same underlying state representation at
  its leaves. Worth revisiting later if richer features alone don't close
  the gap.
- No change to reward structure, starvation timeout, or collision rules —
  purely a state-representation and training-budget change.
- No backward compatibility with an existing `q_table.json`. The table
  shape changes from `(72, 3)` to `(1600, 3)`, so an old saved table is
  simply incompatible. Not handled defensively — `q_table.json` is already
  gitignored, and retraining from scratch is the expected path.

## Danger distance (ray-cast + bucket)

Replace the single-step `is_danger` boolean check with a ray-cast that
counts free cells before the first obstacle (wall or non-tail body
segment), then buckets that count:

```python
MAX_DANGER_SCAN = 6

def _ray_distance(head, direction, grid_size, occupied, max_scan=MAX_DANGER_SCAN) -> int:
    cell = head
    for distance in range(max_scan):
        cell = direction.apply(cell)
        in_bounds = 0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size
        if not in_bounds or cell in occupied:
            return distance
    return max_scan


def _danger_bucket(distance: int) -> int:
    if distance == 0:
        return 0  # obstacle adjacent — matches the old boolean True exactly
    if distance <= 2:
        return 1
    if distance <= 5:
        return 2
    return 3  # far / effectively safe
```

`occupied` stays `snake.pos_set - {snake.tail}` — the same tail-exclusion
rule as today (the tail vacates on a non-growth move, so it isn't treated as
an obstacle). This means a ray whose only in-range obstacle is the tail
treats it as passable and keeps scanning past it. The scan early-exits after
`MAX_DANGER_SCAN` steps since resolution beyond the "far" bucket boundary
adds no information — cheap regardless of `grid_size`.

`distance == 0` is bucket 0, an exact match for the old boolean `True`, so
bucket 0 is a strict refinement of the old danger flag; buckets 1-3
subdivide what used to be a single `False`.

## Food distance (bucket the existing magnitude)

The dot-product magnitude already exists in `from_world`'s current
computation — it's just discarded via `Sign.of()`. Keep it:

```python
def _food_bucket(component: int) -> int:
    if component == 0:
        return 0
    magnitude = 1 if abs(component) <= 3 else 2
    return magnitude if component > 0 else -magnitude
```

Result in `{-2, -1, 0, 1, 2}` — same POS/ZERO/NEG structure as today, with
"behind" and "ahead" each split into near (magnitude ≤ 3) and far
(magnitude > 3).

## `SnakeState` and the index formula

```python
N_DANGER_BUCKETS = 4
N_FOOD_BUCKETS = 5  # per axis

@dataclass(frozen=True, slots=True)
class SnakeState:
    dng_straight: int  # 0-3, see _danger_bucket
    dng_right: int
    dng_left: int
    food_fwd: int  # -2..2, see _food_bucket
    food_lat: int

    N_STATES = N_DANGER_BUCKETS ** 3 * N_FOOD_BUCKETS ** 2  # 1600

    @property
    def index(self) -> int:
        danger_component = (
            self.dng_straight * N_DANGER_BUCKETS + self.dng_right
        ) * N_DANGER_BUCKETS + self.dng_left
        food_component = (self.food_fwd + 2) * N_FOOD_BUCKETS + (self.food_lat + 2)
        return danger_component * (N_FOOD_BUCKETS ** 2) + food_component

    @classmethod
    def from_world(cls, snake, food, grid_size) -> "SnakeState":
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

        return cls(dng_straight, dng_right, dng_left, food_fwd, food_lat)
```

This is a mixed-radix positional index (base-4 for the three danger
buckets, base-5 for the two food buckets), generalizing the old 2-bit/3-way
bit-packing to arbitrary bucket counts. `SnakeState.N_STATES` remains the
only thing `QLearningAgent`/`train.py` need to know about — both already
take it as an opaque constructor/module value, so nothing downstream of
`SnakeState` needs to change to accommodate the larger state space.

## Cleanup: remove `Sign`

`Sign` (in `snake_types.py`) becomes entirely unused once `food_fwd`/
`food_lat` are plain ints — grep confirms it's referenced nowhere outside
`snake_types.py` itself, `snake_state.py` (rewritten by this task), and the
two test files. Delete `Sign` from `snake_types.py` and its dedicated
`TestSign` class from `tests/test_snake_types.py`.

## Training defaults

The state space grows 1600/72 ≈ 22x. The same training budget (5000-episode
epsilon decay, 10000 total episodes) that converged the 72-state table
won't adequately visit most of the 1600 states. Bump both defaults:

- `QLearningAgent.__init__`: `epsilon_decay_episodes` default `5000` → `100_000`
- `train()`: `n_episodes` default `10000` → `200_000`

Training remains fast in absolute terms (~6.4s per 10k episodes previously),
so 200k episodes should stay well under a couple of minutes.

## Testing plan

- `tests/test_snake_types.py`: remove `TestSign` and the now-unused `Sign`
  import. `TestAction`/`TestDirection` unaffected.
- `tests/test_snake_state.py` (substantial rewrite):
  - `TestIndex`: new cases for the mixed-radix formula — danger buckets
    scale by `N_FOOD_BUCKETS ** 2` (25) per unit, food buckets offset
    correctly across the full `{-2..2} x {-2..2}` grid, and an exhaustive
    combinatorial check that all 1600 combinations produce unique indices
    in `[0, 1600)`.
  - `TestImmutability`: same frozen-dataclass check, updated to construct a
    state with int fields instead of `Sign`.
  - `TestFromWorldDanger` (rewritten): wall-adjacent → bucket 0 (all four
    edges); non-tail body adjacent → bucket 0; obstacle at distance 1 and 2
    → bucket 1; obstacle at distance 3 and 5 → bucket 2; open space
    (nothing within `MAX_DANGER_SCAN`) → bucket 3; tail-exclusion still
    holds when the tail is the only in-range obstacle (ray continues past
    it to whatever's beyond, or bucket 3 if nothing).
  - `TestFromWorldFoodBuckets` (rewritten): food at each of the four
    cardinal offsets, at both a near distance (≤3) and a far distance (>3),
    for each of the four headings.
- `tests/test_q_agent.py`/`tests/test_train.py`: unaffected — both only
  ever consume `SnakeState.N_STATES` as an opaque integer, never
  `SnakeState`'s internal fields.
