# Q-Learning Agent — Design

## Purpose

Add a tabular Q-learning agent that trains against the existing `SnakeEnv`,
plus the training loop that runs episodes and reports progress. Includes one
small preliminary refactor to `SnakeEnv.step()`'s return contract, motivated
by keeping the training loop (and any future consumer) resistant to shape
changes.

## Non-goals

- No pygame / rendering code of any kind. See "Accounting for the future
  renderer" below for why the design doesn't need to do anything special to
  accommodate one later.
- No neural-network / function-approximation agent — tabular only, backed by
  a plain Python nested list, matching `SnakeState.N_STATES == 72`.
- No CLI argument parsing for hyperparameters — `train()` takes plain
  keyword arguments with defaults; if configurability beyond that is needed
  later, that's a separate follow-up.

## Preliminary refactor: `StepResult` dataclass

### Motivation

`SnakeEnv.step()` currently returns a bare `tuple[SnakeState, float, bool,
dict]`, unpacked positionally at every call site
(`state, reward, done, info = env.step(action)`). Two problems with this
going into the Q-learning work:

1. Positional tuples are fragile to refactor — any future change to the
   shape (adding a field, reordering) silently breaks every unpacking call
   site with no type error.
2. The starvation-timeout-vs-death distinction currently lives in
   `info["truncated"]`, a key that's only present some of the time. Callers
   already have to remember `info.get("truncated", False)` — exactly the
   kind of easy-to-typo, easy-to-forget pattern the dataclass is meant to
   eliminate.

### Design

`snake_env.py` gains a `StepResult` dataclass, replacing `step()`'s bare
tuple return:

```python
@dataclass(frozen=True, slots=True)
class StepResult:
    state: SnakeState
    reward: float
    done: bool
    truncated: bool
    info: dict  # currently only ever {"score": snake.length}
```

`step()`'s signature changes from `-> tuple[SnakeState, float, bool, dict]`
to `-> StepResult`. Behavior is unchanged — this is a return-shape refactor,
not a logic change:

- Collision path: `StepResult(state, DEATH_REWARD, True, False, {"score": snake.length})`
- Normal/starvation path: `StepResult(state, reward, done, done, {"score": snake.length})`
  — `truncated` and `done` are the same value here because the collision
  path (the only other way `done` becomes `True`) already returned above,
  so reaching this line at all means the only source of `done=True` is the
  starvation timeout.

`reset()` is unchanged (`-> SnakeState`) — a single return value was never
fragile in the same way.

### Migration

- `snake_env.py`: add the `StepResult` dataclass, change `step()`'s return
  statements and type annotation.
- `tests/test_snake_env.py`: every existing test that does
  `state, reward, done, info = env.step(...)` or checks `"truncated" in info`
  / `info["truncated"]` needs updating to `result = env.step(...)` with
  `result.state` / `result.reward` / `result.done` / `result.truncated` /
  `result.info["score"]`. The lifecycle soak test's per-step invariant
  checks need the same treatment.
- No other file currently consumes `step()` (no training loop exists yet),
  so this migration is contained entirely to `snake_env.py` and its test
  file.

## Accounting for the future renderer

A pygame renderer will eventually need to draw the actual world (snake body
cells, food position, grid dimensions) — data that `SnakeState` deliberately
does *not* carry, since it's a lossy 72-state encoding built for the Q-table,
not for drawing pixels. `SnakeEnv` already exposes what a renderer would
need as plain public attributes: `.snake` (for `.body`), `.food`, and
`.grid_size`. A future renderer module reads those directly after each
`reset()`/`step()` call; `SnakeEnv` never imports pygame, never gains a
`render()` method, and never needs to know a renderer exists. No protocol or
interface is added now — there is nothing to abstract over on the env side,
since the dependency runs renderer → env, not the reverse. If multiple
renderer backends are ever wanted (pygame vs. terminal ASCII), that
polymorphism belongs on the renderer side, not `SnakeEnv`'s.

## `QLearningAgent` (new `q_agent.py`)

Pure agent logic — no dependency on `Snake` or `SnakeEnv`, only on
`SnakeState.N_STATES`-shaped indices and the `Action` enum. This mirrors the
existing project pattern of small, independently testable modules.

```python
class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int = 3, alpha: float = 0.1,
                 gamma: float = 0.9, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01, epsilon_decay_episodes: int = 5000):
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

    def update(self, state_index: int, action: Action, reward: float,
               next_index: int, done: bool, truncated: bool) -> None:
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

Defaults: `alpha=0.1`, `gamma=0.9`, `epsilon_start=1.0`, `epsilon_end=0.01`,
`epsilon_decay_episodes=5000`.

**Bellman target rule** (the reason `StepResult.truncated` exists): on real
death (`done=True, truncated=False`), the target is `reward` alone — the
episode genuinely ended, there is no next state to bootstrap from. On
truncation (`done=True, truncated=True`) and on normal steps
(`done=False`), the target bootstraps as usual:
`reward + gamma * max(q_table[next_index])`. This is what lets a
stalled-but-still-alive state keep a meaningful learned value instead of
being taught "this state is worthless."

**Tie-breaking:** `choose_action`'s greedy branch always picks the
lowest-index action on a tie (e.g. an unvisited state where all three
Q-values are still `0.0`). Deterministic, and irrelevant in practice since
`epsilon` starts at `1.0` (mostly-random) during exactly the period when
most states are still all-zero.

## Training loop (new `train.py`)

```python
def train(n_episodes: int = 10000, grid_size: int = 12,
          save_path: str = "q_table.json") -> QLearningAgent:
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
            agent.update(state.index, action, result.reward,
                         result.state.index, result.done, result.truncated)
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

`main.py` (currently empty) becomes the project's entry point:

```python
from train import train

if __name__ == "__main__":
    train()
```

Running `python main.py` runs a full training run end-to-end and writes
`q_table.json`.

## Testing plan

- `tests/test_snake_env.py` (modify, as part of the preliminary refactor):
  update every existing `step()`-consuming test to use `StepResult`
  attribute access instead of tuple unpacking / `info["truncated"]` lookups.
  No new behavior to test — this is a mechanical migration.
- `tests/test_q_agent.py` (new):
  - `choose_action` with `epsilon=0` always returns the greedy (argmax) action
  - `choose_action` with `epsilon=1` always explores (statistical, over many calls)
  - tie-breaking picks the lowest index on an all-zero row
  - `update()`'s Bellman math for a normal (non-terminal) step
  - `update()` on real death (`done=True, truncated=False`) uses `target=reward` only
  - `update()` on truncation (`done=True, truncated=True`) bootstraps like a non-terminal step
  - `save()`/`load()` round-trips the table exactly (via a temp file)
- `tests/test_train.py` (new): run `train()` with a small `n_episodes`
  (e.g. 20) and small `grid_size`, assert it completes without error,
  returns a `QLearningAgent` whose `q_table` has `SnakeState.N_STATES` rows
  of length 3, and that `agent.epsilon` decreased from its start value.
  This exercises the real env with randomness, so it stays a small
  sanity/shape check, not a check of learned quality.
