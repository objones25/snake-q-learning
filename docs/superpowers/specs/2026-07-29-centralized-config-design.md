# Centralized Config Design

## Problem

Parameters are scattered and duplicated across the codebase instead of living in one place:

- `grid_size=20` is redeclared as a default in `SnakeEnv.__init__`, `train()`, `play()`, `main.py`'s two subparsers, and `watch.py`'s two subparsers (7 places).
- `n_episodes` defaults (`30_000` for train, `100` for play) are duplicated between `main.py`'s parser, `train.py`/`play.py`'s signatures, and `watch.py`'s parser.
- `QLearningAgent`'s learning hyperparameters (`alpha`, `gamma`, `epsilon_start`, `epsilon_end`, `epsilon_decay_episodes`, `n_actions`) have no CLI exposure at all — tuning them requires editing `q_agent.py`'s constructor defaults directly.
- `render_every`/`cell_size`/`fps` are duplicated between `watch.py`'s two subparsers and its two `watch_*` functions.

Every function in the `main.py`/`watch.py` → `train.py`/`play.py` → `QLearningAgent`/`SnakeEnv`/`PygameRenderer` chain re-declares the same parameters instead of receiving one injected object.

## Goals

- One source of truth per parameter's default value.
- Preserve the existing strict layering (`CLAUDE.md`): `QLearningAgent` and `PygameRenderer` each depend only on the config that describes their own concerns, not a monolithic god-object.
- No behavior change — every default value stays numerically identical to today.
- CLI surface grows (new flags for agent hyperparameters) but every currently-working invocation of `main.py`/`watch.py` keeps working with the same flags.

## Non-goals

- No change to `SnakeEnv`'s constructor (`grid_size: int` stays a plain scalar — it's a single field, not worth its own config type) or to `Snake`/`SnakeState`/`snake_types.py`.
- No backwards-compatibility shim for the old scalar-kwarg signatures of `train()`/`play()`/`QLearningAgent`/`PygameRenderer` — this is a breaking internal refactor, not a public API with external consumers.

## Design

### New module: `config.py`

Sits at the bottom of the dependency graph, alongside `snake_types.py` — pure frozen dataclasses, no imports from the rest of the codebase.

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

`n_actions` is a config field but deliberately gets **no CLI flag** in `main.py`/`watch.py` — it's tied 1:1 to the 3-member `Action` enum, and exposing it as a tunable would let someone desync the Q-table shape from the actual action space.

`PlayConfig` does **not** carry an `AgentConfig` field. `play()` never learns (epsilon is forced to 0, `agent.update` is never called), so there's nothing for a caller to usefully override; `play()` constructs its agent with a bare default `AgentConfig()` internally.

No `WatchTrainConfig`/`WatchPlayConfig` wrapper type — `watch_train`/`watch_play` each take their constituent configs as separate parameters instead of introducing a new type that would serve exactly one call site each.

### Constructor / function signature changes

- **`QLearningAgent.__init__(self, n_states: int, config: AgentConfig = AgentConfig())`** replaces the current 6 scalar kwargs (`n_actions`, `alpha`, `gamma`, `epsilon_start`, `epsilon_end`, `epsilon_decay_episodes`). Reads all six off `config`.
- **`PygameRenderer.__init__(self, grid_size: int, config: RenderConfig = RenderConfig())`** — `grid_size` stays a plain scalar (shared board concept with `SnakeEnv`, not a rendering concept); `cell_size`/`fps` come from `config`.
- **`train(config: TrainConfig = TrainConfig()) -> Iterator[EpisodeStep]`** replaces `n_episodes`/`grid_size`/`save_path`. Builds `SnakeEnv(grid_size=config.grid_size)` and `QLearningAgent(SnakeState.N_STATES, config.agent)`.
- **`play(config: PlayConfig = PlayConfig()) -> Iterator[EpisodeStep]`** replaces `n_episodes`/`grid_size`/`q_table_path`. Builds `QLearningAgent(SnakeState.N_STATES, AgentConfig())` internally (default hyperparams — irrelevant post-load since epsilon is forced to 0).
- **`watch_train(train_config: TrainConfig = TrainConfig(), render_config: RenderConfig = RenderConfig(), render_every: int = 1) -> None`** — `render_every` stays a plain scalar; it's specific to `watch_train`, not shared with `watch_play`.
- **`watch_play(play_config: PlayConfig = PlayConfig(), render_config: RenderConfig = RenderConfig()) -> None`**

### CLI wiring (`main.py`, `watch.py`)

Flag names are unchanged for everything that exists today (`--n-episodes`, `--grid-size`, `--save-path`, `--q-table-path`, `--render-every`, `--cell-size`, `--fps`). New flags are added to the `train` subcommands only (both `main.py` and `watch.py`), for the previously CLI-inaccessible agent hyperparameters: `--alpha`, `--gamma`, `--epsilon-start`, `--epsilon-end`, `--epsilon-decay-episodes`.

Argparse defaults are read off dataclass instances instead of being re-typed as literals:

```python
_train_defaults = TrainConfig()
_agent_defaults = _train_defaults.agent
train_parser.add_argument("--n-episodes", type=int, default=_train_defaults.n_episodes)
train_parser.add_argument("--grid-size", type=int, default=_train_defaults.grid_size)
train_parser.add_argument("--save-path", type=Path, default=_train_defaults.save_path)
train_parser.add_argument("--alpha", type=float, default=_agent_defaults.alpha)
train_parser.add_argument("--gamma", type=float, default=_agent_defaults.gamma)
train_parser.add_argument("--epsilon-start", type=float, default=_agent_defaults.epsilon_start)
train_parser.add_argument("--epsilon-end", type=float, default=_agent_defaults.epsilon_end)
train_parser.add_argument(
    "--epsilon-decay-episodes", type=int, default=_agent_defaults.epsilon_decay_episodes
)
```

`_run_train`/`_run_play` (in `main.py`) and `main()` (in `watch.py`) assemble a `TrainConfig`/`PlayConfig` (and `AgentConfig`/`RenderConfig` where applicable) from parsed `args`, then call `train(config)`/`play(config)`/`watch_train(...)`/`watch_play(...)` with the object(s) instead of a growing kwarg list.

### Non-obvious implementation notes

- A frozen dataclass instance as a function default arg (`config: AgentConfig = AgentConfig()`) is safe from the mutable-default-argument foot-gun since it can't be mutated after construction. `field(default_factory=...)` is still required (and used) for `TrainConfig.agent`, since that's a dataclass field, not a function parameter — Python dataclasses reject mutable-looking defaults there even when frozen.
- `renderer.py` and `q_agent.py` gain a new `import config` dependency. This is intentional: `config.py` is a shared bottom-layer module, like `snake_types.py`, sitting *below* both — not a peer module they'd otherwise have no reason to know about.

## Testing impact

This is a breaking signature change; existing tests are rewritten, not just extended:

- `tests/test_q_agent.py`: construction becomes `QLearningAgent(n_states, AgentConfig(alpha=...))`.
- `tests/test_train.py`, `tests/test_play.py`: calls become `train(TrainConfig(n_episodes=20, grid_size=8, save_path=path))` / `play(PlayConfig(...))`. The `inspect.signature(train).parameters["n_episodes"].default == 30_000` style assertions are replaced with direct dataclass-default assertions (`TrainConfig().n_episodes == 30_000`).
- `tests/test_main.py`: the `fake_train(**kwargs)` spy pattern is replaced with a spy that captures the single `config` positional argument and asserts against its fields (or compares whole config objects for equality, which frozen dataclasses support for free).
- New `tests/test_config.py`: pins every default value to today's numbers (`epsilon_decay_episodes=5_000`, `n_episodes=30_000` for train / `100` for play, `grid_size=20`, etc. — guarding the CLAUDE.md-documented "deliberately reverted" defaults from silently drifting) and asserts each config is frozen.

No behavior change is intended anywhere in this refactor — only where parameters live and how they're threaded through.
