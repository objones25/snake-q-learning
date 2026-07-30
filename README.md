# snake-q-learning

A tabular Q-learning agent that learns to play Snake. No neural network, no
function approximation — the agent's entire "brain" is a `list[list[float]]`
Q-table indexed by a hand-designed discrete state encoding, updated with a
plain Bellman-equation rule. The environment is a small Gym-like world
(`reset()` / `step()`) built from scratch on top of a minimal `Snake` entity.

The interesting part of this project isn't the Q-learning algorithm itself
(textbook tabular Q-learning, nothing novel) — it's the state representation
that makes a 1,600-row table workable at all, and a few structural decisions
(how `Snake` is decoupled from the game world, and why training/inference are
generators rather than functions) that came out of iterating on the design
more than once and are easy to get wrong if you don't know why they're there.
The sections below explain those decisions.

## Quick start

Package manager is [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                                    # install deps
uv run python main.py train                # train with all defaults (30,000 episodes, 20x20 grid)
uv run python main.py train --n-episodes 5000 --grid-size 10 --save-path other.json
uv run python main.py play                 # load q_table.json, play 100 episodes greedily (epsilon=0)
uv run python main.py play --n-episodes 10 --q-table-path other.json

uv sync --extra render                     # install optional pygame dependency
uv run --extra render python watch.py play --q-table-path q_table.json
uv run --extra render python watch.py train --render-every 500

uv sync --extra plot                       # install optional matplotlib dependency
uv run --extra plot python main.py train --plot --plot-path training.png
uv run --extra plot python main.py play --plot --plot-path scores.png

uv sync --extra api                        # install optional fastapi/uvicorn dependencies
uv run --extra api uvicorn api:app --reload# serve /train and /play as SSE streams on :8000

uv run pytest                              # full test suite
```

`train` runs episodes, learns, and dumps a Q-table to `q_table.json`. `play`
loads a saved Q-table and runs the agent greedily (`epsilon=0`, no further
learning). `watch.py` is a separate, entirely optional entry point that
renders either of those with a pygame window; `pygame` is never a hard
dependency — headless training and play never import it. `--plot` on
`train`/`play` works the same way with matplotlib: plots the rolling-average
training-score curve, or the score distribution across a play run,
respectively; omit `--plot-path` to open an interactive window instead of
saving to a file.

`api.py` is a separate, optional HTTP entry point (`uv sync --extra api`) that
exposes `train()`/`play()` as Server-Sent Events over `GET /train` and
`GET /play` — for driving a browser-rendered snake game (e.g. deployed to
Railway via the included `Procfile`). `/train` streams live training,
`/play` always plays back the committed `example_q_table.json` (the real
`q_table.json` is gitignored, so a fresh deployment has nothing else to
load). Neither endpoint writes to disk. See [API.md](API.md) for the full
endpoint reference (params, response format, curl/browser examples).

## Performance

On the default 20x20 grid, running a Q-table trained with the flood-fill
safety shield enabled (`safety.py` — see Architecture below) greedily
(`main.py play`, `epsilon=0`, no further learning, shield still on since
`play()` defaults to `use_shield=True`) over 1,000 episodes:

```
$ uv run main.py play --n-episodes 1000 --q-table-path example_q_table.json
...
avg_score=79.00  top_score=123
```

`score` is the snake's final length — it starts at length 1, so an average
score of ~79 means roughly 78 food items eaten before dying or hitting the
starvation timeout. Per-episode scores vary widely (30s up into the 120s)
since food placement and starting heading are randomized every episode,
even though the greedy policy itself is deterministic per state — running
more play episodes just gives more chances to see an outlier score, it
doesn't change the underlying policy. Use `--plot` (see Quick start) to see
the full score distribution or the training curve behind numbers like
these.

This is roughly double the ~42 average score an earlier, unshielded agent
reached — the shield prevents the agent from wasting exploration on moves
that trap itself, so it learns a meaningfully better policy. A Q-table
trained with the shield on is calibrated for shielded play: running
`--no-shield` (or `use_shield=false` on the API) against a shield-trained
table gives a materially worse policy, not just "the same agent without a
seatbelt."

## Architecture & design decisions

The module layering is strict, each one only knowing about the layer below
it:

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
train.py / play.py  (generators yielding EpisodeStep)
      ↑
  main.py       (CLI: consumes the generators, does the printing)
```

`config.py` and `episode_step.py` sit off to the side as shared leaf modules
(more on both below). A separate optional branch — `renderer.py` /
`watch.py` — renders gameplay by driving `train()`/`play()`'s generators; it
depends on `snake_env.py`'s `Board` snapshot type but neither `main.py` nor
`train.py`/`play.py` know it exists.

### State representation: why the Q-table has 1,600 rows

`SnakeState` (`snake_state.py`) is what turns the actual game — a snake body,
a food cell, a grid — into a single integer that indexes into the Q-table.
This is the crux of the whole project: pick a bad encoding and the agent has
either too many states to ever visit enough times to learn well, or too few
states to distinguish situations that genuinely call for different actions.

The state is a frozen dataclass with five fields, built via a classmethod
that takes a snapshot of the world rather than holding a live reference to
it:

```python
SnakeState.from_world(snake: Snake, food: tuple[int, int], grid_size: int) -> SnakeState
```

**It's rotation-invariant.** The three "danger" fields are ray-cast
distances _relative to the snake's current heading_ — straight ahead, one
turn right, one turn left — not absolute grid directions like north/south.
Food position is encoded the same way: forward/backward and left/right of
the snake's own facing, not x/y on the grid. The practical effect is that
the agent only ever has to learn one policy, not four rotated copies of the
same policy for whichever way it happens to be facing. A "wall two cells to
my right" state looks identical to the agent whether the snake is actually
facing north or east.

**Danger is distance-bucketed, not boolean.** For each of the three relative
directions, `_ray_distance` scans up to `MAX_DANGER_SCAN = 6` cells ahead
until it hits a wall or a non-tail body segment (the snake's own tail is
excluded from the scan because it vacates on a normal move — by the time the
head could reach it, it's no longer there, unless food was just eaten this
turn, see the note on duplicated collision logic below). The raw distance is
then bucketed by `_danger_bucket` into four values: `0` = immediate
collision, `1` = 1-2 cells away, `2` = 3-5 cells away, `3` = clear. Food
position is bucketed similarly by `_food_bucket` into `{-2, -1, 0, 1, 2}` per
axis — zero means aligned, and the near/far split (magnitude ≤ 3 vs. > 3) is
a signed distinction, not just direction.

**Packing five fields into one index.** `SnakeState.index` uses a mixed-radix
encoding — base 4 for the three danger buckets, base 5 for the two food
buckets — giving `N_STATES = 4**3 * 5**2 = 1600`. This is the _only_ coupling
between `SnakeState` and `QLearningAgent`: `train.py` passes
`SnakeState.N_STATES` into the agent's constructor as the Q-table's row
count, and nothing else about the encoding leaks into the agent, which only
ever sees opaque integers.

**Why this replaced a smaller, simpler encoding.** The original design used
three plain booleans for danger and a 3-way sign (`Sign.NEG`/`ZERO`/`POS`,
since removed from the codebase entirely) for food direction — 72 states
total.
That version trained fine but plateaued hard: average score converged to
roughly 21-22 by episode 5,000-6,000 and never improved further even with
30,000 episodes of training. The root cause was information loss, not
under-training — a boolean "danger ahead" can't distinguish "wall directly in
front of me" from "wall five cells away, plenty of room to turn," and a
3-way food sign can't distinguish "food one cell away" from "food across the
entire board." Both are real, decision-relevant information a bigger table
can represent. Switching to the distance-bucketed encoding (72 → 1,600
states) is what let the agent actually improve past that plateau.

An alternative that was considered and explicitly rejected: multi-step
lookahead/planning instead of richer state features. It was ruled out because
it's a much larger change (needs safe environment-forking/simulation
machinery, and turns per-decision cost from O(1) into O(3^k) for a k-step
search) and doesn't fix the actual root cause — a planner built on top of the
same impoverished state features would still be blind at its leaves.

**A training-budget number that was tried, then deliberately undone.** Going
from 72 to 1,600 states is a ~22x jump, and the state-encoding spec initially
called for scaling the training budget to match: bump
`QLearningAgent.epsilon_decay_episodes` from 5,000 to 100,000 and `train()`'s
`n_episodes` from 30,000 to 200,000. That change was made, then reverted a
commit later after a controlled experiment: the 200,000-episode run only
improved the average score by about 1.3 points over the 30,000-episode run,
for roughly 5x the wall-clock time. The smaller budget stayed as the
out-of-the-box default — `epsilon_decay_episodes=5_000`,
`n_episodes=30_000` — precisely because that trade-off wasn't worth it as a
_default_; a longer run is still just a kwarg (or `--epsilon-decay-episodes`
/ `--n-episodes` flag) away for anyone who wants to spend the extra time. If
you're reading an older version of the design docs and see 100,000 /
200,000, treat those as the road not taken, not the current behavior.

### Snake: a deliberately narrow entity

`Snake` (`snake.py`) is intentionally the least-knowledgeable object in the
codebase. It owns exactly three things:

- `body` — a `deque` of `(x, y)` cells, tail first, head last.
- `pos_set` — a `set` mirror of `body`, kept in sync on every `move()` call,
  used purely so collision/occupancy checks are O(1) instead of an O(n) scan
  of the deque.
- `direction` — the current heading.

Its entire public API is three methods: `turn_right()`, `turn_left()`, and
`move(food_consumed: bool)`. That's it. `Snake` has **zero** knowledge of the
grid size, where the food is, or what counts as game over — all of that
lives one layer up, in `SnakeEnv`. Feeding it `food_consumed` as a bool
argument to `move()` (rather than, say, `Snake` querying the environment
itself, or a `grow()` method vs. a `move()` method) keeps that boundary firm:
`Snake` doesn't decide whether it grows, it's told.

This narrowness is what makes `SnakeState` possible as a pure value type.
Because `Snake` doesn't know about food or grid bounds, and because
`SnakeState.from_world()` takes a `(snake, food, grid_size)` snapshot rather
than holding a live reference into a `Snake` instance, `SnakeState` can be a
plain frozen dataclass with an `.index` property — nothing about it needs to
stay "attached" to a mutable game object. The same discipline shows up again
one layer up: `SnakeEnv.step()`'s `Board` snapshot (see below) does
`tuple(self.snake.body)` specifically so a consumer holding a `Board` isn't
silently looking at stale-then-mutated data after the next `step()` call —
`Snake.body` is a `deque` that gets mutated in place, so anything that needs
to survive past the next move has to be copied out, not referenced.

One consequence worth flagging rather than hiding: collision/tail-exclusion
logic is **duplicated, not shared**, between `SnakeEnv.step()` (which
excludes the tail from the obstacle set unless food was just eaten this
move) and `SnakeState.from_world()`'s danger ray-casting (which
unconditionally excludes the tail). They agree in practice, but nothing
enforces that they stay in sync if one side changes without the other — it's
a place to double-check when touching collision or danger-sensing code.

### `train()` / `play()` as generators, not functions

`train.py` and `play.py` don't return a final result, and they don't print
anything. Both are generator functions that yield an `EpisodeStep` after
every single environment step (not just at episode boundaries):

```python
@dataclass(frozen=True, slots=True)
class EpisodeStep:
    episode: int
    result: StepResult   # state, reward, done, truncated, info, board
    agent: QLearningAgent
```

`play()` mirrors `train()`'s loop almost exactly, but never calls
`agent.update` — pure greedy inference, no learning happening at all.
Forcing `epsilon = 0.0` isn't `play()`'s own job anymore, either: both
`train()` and `play()` now take an already-constructed `env`/`agent`
rather than building or loading either themselves (see the config section
below), so each caller — `main.py`'s `_run_play`, `watch.py`'s
`watch_play`, `api.py`'s `_stream_play` — loads the Q-table and sets
`agent.epsilon = 0.0` itself before calling `play()`.

This shape wasn't the original design — both functions used to run their
loop internally, print progress as they went, and `train()` returned the
final trained agent. They were converted into generators specifically to
support `watch.py`, a pygame renderer that needs to observe the _same_
step-by-step stream that drives the CLI's progress printing, without
duplicating the training or inference loop a second time. Concretely: before
the conversion, adding a renderer would have meant either copy-pasting
`train()`'s episode loop into a render-aware variant (and then keeping two
copies of the same logic in sync forever), or bolting an optional
render-callback parameter onto `train()`/`play()` themselves, dragging a
rendering concern down into a module that has no business knowing pygame
exists. Converting both to generators sidesteps the problem entirely:
`main.py` and `watch.py` both just iterate the same generator, and each
consumer decides for itself what to do with each `EpisodeStep` —
`main.py`'s `_run_train`/`_run_play` accumulate scores and print a progress
line every 500 episodes; `watch.py` feeds `step.result.board` to
`PygameRenderer.draw()` and (for training) samples every Nth episode via
`--render-every` instead of rendering all 30,000.

Two details fall out of this that are easy to get bitten by:

- **A generator's body doesn't run until it's iterated.** Calling
  `train(env, agent, n_episodes)` or `play(env, agent, n_episodes)` returns
  an iterator immediately — none of the loop body runs, not even the first
  `env.reset()`, until something actually starts consuming it (`next(...)`,
  a `for` loop, `list(...)`). `main.py`'s `_run_train`/`_run_play` and
  `watch.py`'s `watch_train`/`watch_play` all do this with a plain `for`
  loop. An earlier version of `play()` leaned on this directly — its
  `FileNotFoundError` check for a missing Q-table path sat before the first
  `yield` inside `play()` itself, so it only fired once iteration started —
  but that check has since moved out to each caller, which now loads the
  Q-table (and can fail fast on a missing path) before calling `play()` at
  all, rather than `play()` doing any loading itself.
- **`board` vs. `state` are not the same thing, and only one of them is
  lossy.** `StepResult.board` (a `Board`: `grid_size`, `snake_body`, `food`
  as raw coordinates) exists purely so `watch.py`'s renderer has something
  to draw pixels from. `StepResult.state` (a `SnakeState`) is the
  rotation-invariant, heavily bucketed encoding described above — it cannot
  be used to reconstruct where anything actually is on the board. They're
  built by two separate calls inside `SnakeEnv.step()` for exactly this
  reason: one is for the Q-table, one is for drawing a window.

`EpisodeStep` itself lives in its own small module, `episode_step.py`,
rather than inside `snake_env.py`. Putting it in `snake_env.py` would make
the env layer import `QLearningAgent`, inverting the whole bottom-up
layering described above; `episode_step.py` instead sits at the same top
layer as `train.py`/`play.py`, which already depend on both `snake_env` and
`q_agent` directly.

### Supporting design decisions

A few smaller decisions underpin the above and are worth knowing about, if
not worth the same depth:

- **`StepResult` is a frozen dataclass, not a tuple.** `SnakeEnv.step()`
  originally returned a bare `tuple[SnakeState, float, bool, dict]`, unpacked
  positionally at every call site. That's fragile — any future change to the
  shape (new field, reordering) silently breaks unpacking with no type
  error, and the starvation-vs-death distinction used to live in an
  inconsistently-present `info["truncated"]` key. `StepResult` (`state,
reward, done, truncated, info, board`) replaced it. The `done`/`truncated`
  split matters for learning correctness: `done=True` covers both real death
  (wall/self collision) and starvation truncation
  (`steps_since_food > 100 * snake.length`), but `QLearningAgent.update`
  only skips Bellman bootstrapping (`target = reward` alone) when `done and
not truncated` — real death. On truncation, it still bootstraps
  (`reward + gamma * max(q_table[next_index])`), because a
  stalled-but-still-alive state shouldn't be taught it's worthless the way
  actual death should be. Getting this condition backwards is a subtle bug
  that silently degrades learning quality rather than crashing anything.
- **Rewards are sparse and unshaped**: `FOOD_REWARD = 10`, `DEATH_REWARD =
-10`, `STEP_REWARD = 0`. No distance-based reward shaping — the agent has
  to learn to route toward food using only the bucketed state features and
  the terminal food/death signals.
- **`config.py` centralizes every scattered parameter.** Learning
  hyperparameters, grid size, episode counts, render settings, and file
  paths used to be redeclared as defaults in half a dozen places (`SnakeEnv`,
  `train()`, `play()`, `main.py`'s subparsers, `watch.py`'s subparsers).
  `AgentConfig` / `RenderConfig` / `TrainConfig` / `PlayConfig` are frozen
  dataclasses that now give each of those parameters exactly one source of
  truth, consumed directly by `QLearningAgent` and `PygameRenderer`.
  `train()`/`play()` themselves don't take a `TrainConfig`/`PlayConfig` —
  their signature is `(env, agent, n_episodes, use_shield=True)` — so
  `main.py`, `watch.py`, and `api.py` each build a config object from
  parsed CLI args (or, for `api.py`, query params), then construct the
  `env`/`agent` from it before calling `train()`/`play()`, instead of
  threading a growing list of scalar kwargs through every layer.

## Testing

Plain `pytest`, no `unittest.TestCase`, organized as `Test*` classes grouped
by behavior, heavy use of `@pytest.mark.parametrize`. Notably,
`tests/test_snake_env.py::TestLifecycle::test_random_policy_episodes_hold_invariants_every_step`
is a 500-episode soak test that checks environment invariants on every
single step — worth rerunning after any change to `Snake`, `SnakeState`, or
`SnakeEnv`, since it's what previously caught a `pos_set` desync bug in
`Snake.move`.
