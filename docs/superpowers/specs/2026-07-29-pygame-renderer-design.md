# Pygame Renderer — Design

## Purpose

Add a pygame-based visual renderer that can watch either a live training run
or a loaded agent playing greedily, without coupling `train.py` or `play.py`
to pygame (or to rendering at all). Getting there requires two small
refactors first: giving `StepResult` enough raw data to draw a frame, and
turning `train()`/`play()` into generators so a renderer can consume the same
per-step data a CLI printer consumes, without either duplicating the other's
loop.

## Non-goals

- No terminal/ASCII renderer, no multiple renderer backends — pygame only.
- No pause/rewind/speed-slider UI inside the renderer — just draw, quit on
  window close, and a `--render-every` sampling knob for `watch train` so you
  don't have to render all 30,000 training episodes.
- No change to collision detection, reward logic, `reset()`, or any other
  part of `SnakeEnv`'s game rules. This is purely an additive read path.
- No attempt to render the very first frame of an episode (the position
  right after `reset()`, before any action). `reset()` returns a bare
  `SnakeState`, not a `StepResult`, and isn't part of the yielded stream —
  see "Known limitations" below.

## Supersedes: the previous renderer plan

`docs/superpowers/specs/2026-07-27-q-learning-agent-design.md`'s "Accounting
for the future renderer" section assumed a future renderer would read
`SnakeEnv`'s public attributes (`.snake`, `.food`, `.grid_size`) directly,
with the dependency running renderer → env. That plan is superseded here:
instead, `SnakeEnv` produces an immutable snapshot (`Board`) as part of
`StepResult`, and neither `watch.py` nor `renderer.py` ever imports or
touches `SnakeEnv` or `Snake` directly. The reasoning that changed this:
holding a live reference into `env.snake.body` means the data changes out
from under any consumer that doesn't read it before the very next `step()`
call, and threading `env` itself through the new `EpisodeStep` type would
force that type to reach across two dependency layers at once. A snapshot
avoids both problems.

## `Board` and `StepResult.board` (`snake_env.py`)

```python
@dataclass(frozen=True, slots=True)
class Board:
    grid_size: int
    snake_body: tuple[tuple[int, int], ...]
    food: tuple[int, int]
```

`StepResult` gains a fifth field, `board: Board`. `SnakeEnv` gains:

```python
def render_state(self) -> Board:
    return Board(grid_size=self.grid_size, snake_body=tuple(self.snake.body), food=self.food)
```

called at both existing return points inside `step()` (the early collision
return and the normal/starvation return), alongside the existing
`SnakeState.from_world(...)` call. `reset()` is unchanged — it still returns
a bare `SnakeState`.

**Why sparse coordinates, not a dense grid:** the grid is always square, so
`grid_size: int` fully describes it (matching every other place in the
codebase). A dense `grid_size × grid_size` array would allocate 400 cells
per step at the default grid size regardless of how short the snake is;
`snake_body` as a tuple only costs as much as the snake's actual length.

**Why a real copy, not a live reference:** `tuple(self.snake.body)` costs
O(snake length) per step, but `step()` already pays that same order of cost
twice per step today (building `occupied = snake.pos_set - {snake.tail}` in
`SnakeEnv.step()`, and again inside `SnakeState.from_world`) — so this is a
proportionate third instance of existing work, not a new order of magnitude.
In exchange, `Board` is a genuine immutable snapshot: nothing further mutates
it once yielded, unlike a bare reference to `snake.body` (a `deque` that
`Snake.move()` mutates in place on the very next step). No opt-in flag to
skip building it — the marginal cost doesn't justify threading a flag through
`step()`'s callers, and this project has already moved away from
flag/enum-driven dispatch for small APIs.

## `EpisodeStep` (new `episode_step.py`)

```python
@dataclass(frozen=True, slots=True)
class EpisodeStep:
    episode: int
    result: StepResult
    agent: QLearningAgent
```

Lives in its own module rather than `snake_env.py`, because embedding
`QLearningAgent` there would make the env layer import the agent layer,
inverting this project's strict bottom-up layering
(`snake_types → snake → snake_state → snake_env → q_agent → train/play`).
`episode_step.py` sits at the same top layer as `train.py`/`play.py`, which
already both depend on `q_agent` and `snake_env` directly — so this is a
shared leaf module at that layer, not a new layering violation. No `env`
field: everything a consumer needs to render (`result.board`) or report
(`result.info["score"]`) is already on `StepResult`; `agent` stays because
`train`'s CLI printer needs `agent.epsilon`.

## Generator conversion: `train.py` / `play.py`

Both functions keep their existing loop bodies and signatures, but:

- every `print(...)` call is deleted
- every `env.step()` call is followed by
  `yield EpisodeStep(episode, result, agent)`
- neither function `return`s a value anymore — the final trained `agent`
  (for `train`) or the per-episode scores (for `play`) are recoverable from
  the stream of yielded `EpisodeStep`s themselves, since the same live
  `agent` object is yielded every step
- `train()` still ends with `agent.save(save_path)` as the last statement in
  its loop body, after the `for episode in range(n_episodes)` loop completes
  and before the generator's implicit `StopIteration`

**Generator gotcha worth remembering:** none of a generator function's body
executes until it's first iterated — not even code before the first `yield`.
`play()`'s existing `if not q_table_path.exists(): raise FileNotFoundError`
check sits before any `yield`, so simply calling `play(...)` no longer
raises; a caller (or test) has to start iterating first (`next(...)` or
draining it) before that check fires. Both `train.py`'s and `play.py`'s
`if __name__ == "__main__":` guards need to actually drain the generator now
(e.g. `for _ in train(): pass`), or running either script directly would
silently do nothing.

## `main.py`: consuming the generators, owning printing

`main.py` gains two private helpers that are a straight relocation of
`train()`'s/`play()`'s current print logic — not a redesign of it — onto the
`EpisodeStep` stream:

```python
def _run_train(n_episodes, grid_size, save_path):
    recent_scores = deque(maxlen=500)
    top_score = 0
    for step in train(n_episodes=n_episodes, grid_size=grid_size, save_path=save_path):
        if not step.result.done:
            continue
        recent_scores.append(step.result.info["score"])
        top_score = max(top_score, step.result.info["score"])
        if step.episode % 500 == 0:
            avg = sum(recent_scores) / len(recent_scores)
            print(f"episode {step.episode:6d}  epsilon={step.agent.epsilon:.3f}  "
                  f"avg_score={avg:.2f}  top_score={top_score}")


def _run_play(n_episodes, grid_size, q_table_path):
    scores = []
    for step in play(n_episodes=n_episodes, grid_size=grid_size, q_table_path=q_table_path):
        if not step.result.done:
            continue
        score = step.result.info["score"]
        scores.append(score)
        print(f"episode {step.episode:6d}  score={score}")
    avg_score = sum(scores) / len(scores)
    print(f"avg_score={avg_score:.2f}  top_score={max(scores)}")
```

`main()`'s dispatch calls `_run_train(...)`/`_run_play(...)` instead of
`train(...)`/`play(...)` directly. `deque` needs importing into `main.py`.

## `watch.py` + `renderer.py` (new modules)

`renderer.py` is the only file that imports `pygame`. It knows about pixels
and `Board`; nothing about episodes, generators, or agents. Colors are
plain module-level constants — a classic-arcade palette, nothing
configurable: `BG_COLOR = (0, 0, 0)` (black), `SNAKE_COLOR = (0, 200, 0)`
(green), `FOOD_COLOR = (200, 0, 0)` (red).

```python
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

    def _rect(self, cell: tuple[int, int]) -> "pygame.Rect":
        x, y = cell
        return pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)

    def close(self) -> None:
        pygame.quit()
```

`watch.py` is pure glue — it never calls `agent.choose_action` or `env.step`
itself, only drives `train()`'s/`play()`'s existing generators:

```python
def watch_train(n_episodes=30_000, grid_size=20, save_path=Path("q_table.json"),
                 render_every=1, cell_size=24, fps=15):
    renderer = PygameRenderer(grid_size=grid_size, cell_size=cell_size, fps=fps)
    try:
        for step in train(n_episodes=n_episodes, grid_size=grid_size, save_path=save_path):
            if step.episode % render_every != 0:
                continue
            if not renderer.draw(step.result.board, step.episode, step.result.info["score"]):
                break
    finally:
        renderer.close()


def watch_play(n_episodes=100, grid_size=20, q_table_path=Path("q_table.json"),
                cell_size=24, fps=15):
    renderer = PygameRenderer(grid_size=grid_size, cell_size=cell_size, fps=fps)
    try:
        for step in play(n_episodes=n_episodes, grid_size=grid_size, q_table_path=q_table_path):
            if not renderer.draw(step.result.board, step.episode, step.result.info["score"]):
                break
    finally:
        renderer.close()
```

`watch.py` has its own tiny `argparse` CLI (`watch.py train ...` /
`watch.py play ...`), mirroring `main.py`'s subcommand style but entirely
separate — `main.py` never imports `pygame` or `watch.py`, and `watch.py`
never touches `main.py`.

`render_every` exists only on `watch_train`, not `watch_play`: training runs
up to 30,000 episodes and rendering all of them would be both slow and
mostly uninteresting (early episodes are near-random), so it lets you sample
e.g. every 500th episode while ones in between still run at full speed —
the generator underneath doesn't know or care whether it's being rendered.
`watch_play` runs far fewer episodes (100 by default) and is meant to be
watched in full.

## Packaging

`pygame` becomes an optional dependency group in `pyproject.toml`, not a
hard dependency:

```toml
[project.optional-dependencies]
render = ["pygame>=2.5"]
```

`uv sync` never installs pygame; `uv sync --extra render` opts in. A
headless box can run `train.py`/`play.py` and their tests without pygame
ever being importable.

## Error handling

- `watch.py` without the `render` extra installed → plain
  `ImportError: No module named 'pygame'` from `renderer.py`'s top-level
  import. No custom wrapper message — consistent with this codebase's
  practice of not adding validation at non-boundary points.
- `watch_play` against a missing `q_table.json` → propagates the same
  `FileNotFoundError` that `play()` already raises, since `watch_play` just
  iterates `play(...)`.
- Window closed mid-`watch_train` → the `for` loop `break`s. Since
  `train()`'s `agent.save(save_path)` only runs after its episode loop fully
  completes, quitting early means the run's q_table is **not** saved. Watching
  a training run isn't a way to get an early save — let it finish, or run
  headless training via `main.py train` if you want the artifact.
- Window closed mid-`watch_play` → no such downside; `play()` never writes
  to disk.

## Testing plan

- `tests/test_snake_env.py`: unaffected by adding `board` — no existing test
  constructs `StepResult` positionally. New tests added: `result.board`'s
  `grid_size`/`snake_body`/`food` match the env's actual state after a step,
  for both the normal and collision paths.
- `tests/test_train.py`: every test driving `train(...)` to completion needs
  to drain it first — `steps = list(train(...))`, then `agent = steps[-1].agent`
  replaces the old `agent = train(...)`. `test_logs_top_score_alongside_avg_score`
  moves to `tests/test_main.py` (see below), since `train()` itself no longer
  prints anything.
- `tests/test_play.py`: `scores = [s.result.info["score"] for s in play(...) if s.result.done]`
  replaces `scores = play(...)`; the missing-q-table test wraps
  `pytest.raises` around draining the generator, not the bare call;
  `test_prints_per_episode_scores_and_summary` moves to `tests/test_main.py`.
- `tests/test_main.py`: the four existing dispatch tests' monkeypatched fakes
  for `main.train`/`main.play` must return an iterable instead of `None` (the
  real functions are now generators). Two new tests added here, moved from
  `test_train.py`/`test_play.py`, driving the real `train()`/`play()` through
  `main.main([...])` and asserting `top_score=`/`avg_score=` appear in
  captured stdout — this is the end-to-end coverage that the new `_run_train`/
  `_run_play` consumption logic actually works.
- `tests/test_episode_step.py` (new, small): constructing an `EpisodeStep`
  round-trips its three fields.
- `renderer.py`/`watch.py`: no pytest coverage of actual pygame window
  creation or drawing — that's inherently a manual/visual check, not a
  headless-CI-friendly one. After implementation, manually run
  `uv run --extra render python watch.py play` (and `watch.py train` with a
  small `--n-episodes`) to confirm the window opens, the snake/food render
  in the right cells, closing the window stops the loop cleanly, and
  `--render-every` visibly skips episodes during training.

## Known limitations

- The first frame of each episode (the snake's starting position right
  after `reset()`, before any action) is never rendered — `reset()` isn't
  part of the yielded `EpisodeStep` stream. Accepted as a minor cosmetic gap,
  not worth extra plumbing to fix.
- Quitting a `watch_train` window early skips the final `agent.save(...)`
  for that run (see "Error handling").
