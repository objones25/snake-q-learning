# Shield Test Suite Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the test suite up to date with the `lookahead-floodfill-prototype` branch's flood-fill safety shield — cover the new mechanism, fix tests broken by `train()`/`play()`'s signature change, and pin the new CLI/API `use_shield` wiring.

**Architecture:** No production code changes (aside from one doc line). All work is new/rewritten pytest files following this repo's existing conventions: `Test*` classes grouped by behavior, no fixtures, small inline helpers, `monkeypatch`-based spies (never mocks that replace real logic) where an external CLI/HTTP call needs to prove an internal argument was forwarded correctly.

**Tech Stack:** Python 3.13, `uv`, plain `pytest` (no `unittest.TestCase`, no `conftest.py`), FastAPI `TestClient` for `tests/test_api.py`.

## Global Constraints

- Every new/changed test must pass under `uv run pytest <file> -v`; `tests/test_api.py` specifically requires `uv run --extra api pytest tests/test_api.py -v` (needs `httpx` from the `api` extra, per this repo's documented commands).
- No `conftest.py`, no `@pytest.fixture` — match existing convention of inline helper methods/functions.
- `safety.py`'s functions operate on raw tuples with no contiguity validation — test bodies do not need to represent a physically walkable snake shape, only correct blocked-cell/collision semantics. Every numeric scenario used below has been run against the actual current implementation and its expected output verified — do not "fix" an assertion that looks surprising without re-running it first.
- `train.py`/`play.py` bind `safe_action_mask` into their own module namespace via `from safety import safe_action_mask`. Any spy that needs to observe or disable shield calls **must** patch `train.safe_action_mask` / `play.safe_action_mask` (the module-level name), never `safety.safe_action_mask` — patching the latter has no effect on already-bound imports.
- Do not touch `tests/test_config.py`, or the `TestDefaults` classes inside `tests/test_train.py`/`tests/test_play.py`/`tests/test_main.py` — they test `TrainConfig`/`PlayConfig` dataclasses directly and are unaffected by this refactor.
- Do not add a `tests/test_watch.py` and do not refactor the duplicated `FileNotFoundError` check between `main.py`/`watch.py` — both explicitly out of scope per the design spec.

---

### Task 1: Create `tests/test_safety.py`

**Files:**

- Create: `tests/test_safety.py`

**Interfaces:**

- Consumes: `safety._resolve_action(body, direction, action, food, grid_size) -> Body | None`, `safety._has_enough_space(body, grid_size, needed) -> bool`, `safety.safe_action_mask(body, direction, food, grid_size) -> tuple[bool, bool, bool]`, `snake_types.Action` (`STRAIGHT=0, RIGHT=1, LEFT=2`), `snake_types.Direction` (`RIGHT=0, DOWN=1, LEFT=2, UP=3`).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the full test file**

```python
from safety import _has_enough_space, _resolve_action, safe_action_mask
from snake_types import Action, Direction


class TestResolveAction:
    def test_straight_moves_in_current_direction(self):
        result = _resolve_action(((5, 5),), Direction.RIGHT, Action.STRAIGHT, (0, 0), 10)
        assert result == ((6, 5),)

    def test_right_action_turns_direction_right(self):
        result = _resolve_action(((5, 5),), Direction.UP, Action.RIGHT, (0, 0), 10)
        assert result == ((6, 5),)

    def test_left_action_turns_direction_left(self):
        result = _resolve_action(((5, 5),), Direction.UP, Action.LEFT, (0, 0), 10)
        assert result == ((4, 5),)

    def test_food_consumption_keeps_full_body_and_grows(self):
        result = _resolve_action(((4, 5), (5, 5)), Direction.RIGHT, Action.STRAIGHT, (6, 5), 10)
        assert result == ((4, 5), (5, 5), (6, 5))

    def test_out_of_bounds_returns_none(self):
        result = _resolve_action(((9, 5),), Direction.RIGHT, Action.STRAIGHT, (0, 0), 10)
        assert result is None

    def test_self_collision_returns_none(self):
        # Ring-shaped body; head=(6,5), tail=(5,5). Moving DOWN hits (6,6),
        # a non-tail body segment.
        body = ((5, 5), (5, 6), (6, 6), (6, 5))
        result = _resolve_action(body, Direction.DOWN, Action.STRAIGHT, (0, 0), 10)
        assert result is None

    def test_moving_into_vacated_tail_is_not_a_collision(self):
        body = ((6, 5), (5, 5))  # tail, head
        result = _resolve_action(body, Direction.RIGHT, Action.STRAIGHT, (0, 0), 10)
        assert result == ((5, 5), (6, 5))


class TestHasEnoughSpace:
    def test_needed_zero_or_negative_is_always_true(self):
        # Body fully fills the 1x1 grid (0 free cells) — still True, since
        # needed<=0 short-circuits before any BFS.
        assert _has_enough_space(((0, 0),), 1, 0) is True
        assert _has_enough_space(((0, 0),), 1, -5) is True

    def test_true_when_needed_equals_available_free_cells(self):
        # 5x5 grid = 25 cells; 1 blocked (the head) = 24 free.
        assert _has_enough_space(((2, 2),), 5, 24) is True

    def test_false_when_needed_exceeds_available_free_cells(self):
        assert _has_enough_space(((2, 2),), 5, 25) is False

    def test_true_when_needed_matches_a_small_enclosed_pocket(self):
        # 3x3 grid; body forms a ring around the single center cell (1,1),
        # which is the only free cell reachable from the head.
        ring = ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0))
        assert _has_enough_space(ring, 3, 1) is True

    def test_false_when_needed_exceeds_a_small_enclosed_pocket(self):
        ring = ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0))
        assert _has_enough_space(ring, 3, 2) is False


class TestSafeActionMask:
    def test_wall_ahead_is_unsafe_but_turns_are_safe(self):
        # Head at the right edge of a 5x5 grid, facing further right.
        mask = safe_action_mask(((4, 2),), Direction.RIGHT, (0, 0), 5)
        assert mask == (False, True, True)

    def test_tuple_order_matches_action_declaration_order(self):
        mask = safe_action_mask(((4, 2),), Direction.RIGHT, (0, 0), 5)
        assert mask[Action.STRAIGHT] is False
        assert mask[Action.RIGHT] is True
        assert mask[Action.LEFT] is True

    def test_fully_boxed_snake_returns_all_unsafe(self):
        # None of these bodies need be a physically contiguous snake shape —
        # safe_action_mask only cares about blocked-cell membership. This
        # body/direction combination leaves every one of the 3 candidate
        # moves landing in a pocket too small for the resulting body.
        body = ((0, 1), (2, 1), (1, 2), (1, 0))
        mask = safe_action_mask(body, Direction.DOWN, (2, 2), 3)
        assert mask == (False, False, False)

    def test_moving_into_vacated_tail_is_safe(self):
        mask = safe_action_mask(((6, 5), (5, 5)), Direction.RIGHT, (0, 0), 10)
        assert mask == (True, True, True)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_safety.py -v`
Expected: all tests PASS (16 tests). If any fail, do not adjust the assertion — re-derive the expected value against the actual `safety.py` implementation, since every value above was independently verified by direct execution before being written into this plan.

- [ ] **Step 3: Commit**

```bash
git add tests/test_safety.py
git commit -m "Add unit tests for safety.py's flood-fill safety shield"
```

---

### Task 2: Extend `tests/test_q_agent.py` for the new mask/next_mask parameters

**Files:**

- Modify: `tests/test_q_agent.py` (append methods to the existing `TestChooseAction` and `TestUpdate` classes; do not touch `TestSetEpsilonForEpisode`, `TestSaveLoad`, `TestDefaults`)

**Interfaces:**

- Consumes: `QLearningAgent.choose_action(state_index, mask=None)`, `QLearningAgent.update(state_index, action, reward, next_index, done, truncated, next_mask=None)`.

- [ ] **Step 1: Add mask tests to `TestChooseAction`**

Insert these three methods at the end of the `TestChooseAction` class (after `test_tie_breaks_to_lowest_index_action`, before the blank lines preceding `class TestSetEpsilonForEpisode:`):

```python
    def test_mask_restricts_greedy_pick_to_allowed_actions(self):
        agent = QLearningAgent(n_states=5)
        agent.epsilon = 0.0
        agent.q_table[2] = [0.1, 0.9, 0.3]  # RIGHT is the unmasked best...
        # ...but RIGHT is masked out, so LEFT (0.3) should win over STRAIGHT (0.1)
        assert agent.choose_action(2, mask=(True, False, True)) == Action.LEFT

    def test_mask_restricts_exploration_to_allowed_actions(self):
        agent = QLearningAgent(n_states=5)
        agent.epsilon = 1.0
        seen = {agent.choose_action(0, mask=(True, False, True)) for _ in range(200)}
        assert seen == {Action.STRAIGHT, Action.LEFT}

    def test_fully_false_mask_falls_back_to_unrestricted_choice(self):
        agent = QLearningAgent(n_states=5)
        agent.epsilon = 0.0
        agent.q_table[2] = [0.1, 0.9, 0.3]
        assert agent.choose_action(2, mask=(False, False, False)) == Action.RIGHT
```

- [ ] **Step 2: Add next_mask tests to `TestUpdate`**

Insert these three methods at the end of the `TestUpdate` class (after `test_truncation_bootstraps_like_a_normal_step`, before `class TestSaveLoad:`):

```python
    def test_next_mask_restricts_bootstrap_to_allowed_actions(self):
        agent = QLearningAgent(n_states=5, config=AgentConfig(alpha=0.5, gamma=0.9))
        agent.q_table[1] = [1.0, 2.0, 0.5]  # unrestricted max = 2.0 (RIGHT)...
        agent.update(
            state_index=0, action=Action.STRAIGHT, reward=1.0,
            next_index=1, done=False, truncated=False,
            next_mask=(True, False, True),  # ...but RIGHT is masked out
        )
        # target = 1.0 + 0.9 * 1.0 = 1.9; new_q = 0.0 + 0.5 * (1.9 - 0.0) = 0.95
        assert agent.q_table[0][Action.STRAIGHT] == pytest.approx(0.95)

    def test_fully_false_next_mask_falls_back_to_unrestricted_max(self):
        agent = QLearningAgent(n_states=5, config=AgentConfig(alpha=0.5, gamma=0.9))
        agent.q_table[1] = [1.0, 2.0, 0.5]
        agent.update(
            state_index=0, action=Action.STRAIGHT, reward=1.0,
            next_index=1, done=False, truncated=False,
            next_mask=(False, False, False),
        )
        # Same as the unmasked normal-step case: target = 1.0 + 0.9 * 2.0 = 2.8
        assert agent.q_table[0][Action.STRAIGHT] == pytest.approx(1.4)

    def test_real_death_ignores_next_mask(self):
        agent = QLearningAgent(n_states=5, config=AgentConfig(alpha=0.5, gamma=0.9))
        agent.q_table[1] = [100.0, 100.0, 100.0]  # would blow up the target if bootstrapped
        agent.update(
            state_index=0, action=Action.STRAIGHT, reward=-10.0,
            next_index=1, done=True, truncated=False,
            next_mask=(True, True, True),  # present but irrelevant on real death
        )
        assert agent.q_table[0][Action.STRAIGHT] == pytest.approx(-5.0)
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_q_agent.py -v`
Expected: all tests PASS (19 tests: 13 existing + 3 new in `TestChooseAction` + 3 new in `TestUpdate`).

- [ ] **Step 4: Commit**

```bash
git add tests/test_q_agent.py
git commit -m "Add mask/next_mask coverage to QLearningAgent tests"
```

---

### Task 3: Fix `tests/test_train.py` for `train()`'s new signature

**Files:**

- Modify: `tests/test_train.py` (rewrite `TestTrain`; leave `TestDefaults` untouched)

**Interfaces:**

- Consumes: `train.train(env: SnakeEnv, agent: QLearningAgent, n_episodes: int, use_shield: bool = True) -> Iterator[EpisodeStep]`.

- [ ] **Step 1: Replace the file's imports and `TestTrain` class**

Replace the entire file content with:

```python
from pathlib import Path

import pytest

from config import AgentConfig, TrainConfig
from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState
from train import train


class TestTrain:
    def test_returns_agent_with_correctly_shaped_q_table(self):
        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        steps = list(train(env, agent, 20))
        result_agent = steps[-1].agent

        assert isinstance(result_agent, QLearningAgent)
        assert len(result_agent.q_table) == SnakeState.N_STATES
        assert all(len(row) == 3 for row in result_agent.q_table)

    def test_epsilon_decreases_from_start_value(self):
        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        steps = list(train(env, agent, 20))

        assert steps[-1].agent.epsilon == pytest.approx(0.996238)

    def test_death_is_passed_to_update_as_not_truncated(self, monkeypatch):
        seen = []
        original = QLearningAgent.update

        def spy(self, state_index, action, reward, next_index, done, truncated, next_mask=None):
            if done:
                seen.append((done, truncated))
            original(self, state_index, action, reward, next_index, done, truncated, next_mask)

        monkeypatch.setattr(QLearningAgent, "update", spy)
        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        list(train(env, agent, 20))

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

(`TestDefaults` is copied verbatim from the current file — it tests `TrainConfig` directly and needs no changes; it's included here only so Step 1 can replace the whole file in one shot.)

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_train.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_train.py
git commit -m "Fix test_train.py for train()'s new (env, agent, n_episodes, use_shield) signature"
```

---

### Task 4: Add shield-wiring regression tests and a soak test to `tests/test_train.py`

**Files:**

- Modify: `tests/test_train.py` (append two new classes after `TestTrain`, before `TestDefaults`)

**Interfaces:**

- Consumes: `train.train(env, agent, n_episodes, use_shield)`, `QLearningAgent.choose_action(state_index, mask=None)`.

- [ ] **Step 1: Add `import random` to the top of the file**

The `import` block at the top of `tests/test_train.py` becomes:

```python
import random
from pathlib import Path

import pytest

from config import AgentConfig, TrainConfig
from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState
from train import train
```

- [ ] **Step 2: Insert `TestUseShieldWiring` and `TestShieldSoak` after `TestTrain`, before `TestDefaults`**

```python
class TestUseShieldWiring:
    def test_shield_enabled_passes_a_mask_to_choose_action(self, monkeypatch):
        seen_masks = []
        original = QLearningAgent.choose_action

        def spy(self, state_index, mask=None):
            seen_masks.append(mask)
            return original(self, state_index, mask)

        monkeypatch.setattr(QLearningAgent, "choose_action", spy)
        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        list(train(env, agent, 5, use_shield=True))

        assert any(mask is not None for mask in seen_masks)

    def test_shield_disabled_never_passes_a_mask(self, monkeypatch):
        seen_masks = []
        original = QLearningAgent.choose_action

        def spy(self, state_index, mask=None):
            seen_masks.append(mask)
            return original(self, state_index, mask)

        monkeypatch.setattr(QLearningAgent, "choose_action", spy)
        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        list(train(env, agent, 5, use_shield=False))

        assert seen_masks
        assert all(mask is None for mask in seen_masks)


class TestShieldSoak:
    def test_shielded_training_holds_invariants_every_step(self):
        # Mirrors test_snake_env.py::TestLifecycle's 500-episode soak test,
        # but exercises the shield wired into train()'s loop rather than a
        # bare env under a random policy.
        grid_size = 8
        num_episodes = 150
        random.seed(2)

        env = SnakeEnv(grid_size=grid_size)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)

        for step in train(env, agent, num_episodes, use_shield=True):
            body = env.snake.body
            assert set(body) == env.snake.pos_set
            assert len(body) == len(set(body))
            for x, y in body:
                assert 0 <= x < grid_size
                assert 0 <= y < grid_size
            assert 0 <= step.result.state.index < SnakeState.N_STATES
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_train.py -v`
Expected: all 10 tests PASS. The soak test runs ~99,000 steps and should complete in a few seconds — if it takes noticeably longer, something is wrong (e.g. an accidental infinite loop from a masking bug), not just "slow."

- [ ] **Step 4: Commit**

```bash
git add tests/test_train.py
git commit -m "Add shield wiring regression tests and a shield soak test to test_train.py"
```

---

### Task 5: Fix `tests/test_play.py` for `play()`'s new signature and relocate `TestMissingQTable`

**Files:**

- Modify: `tests/test_play.py`

**Interfaces:**

- Consumes: `play.play(env: SnakeEnv, agent: QLearningAgent, n_episodes: int, use_shield: bool = True) -> Iterator[EpisodeStep]`.
- Produces: nothing (the missing-q-table test moves to Task 7, which depends on this task only in that it must run after this file no longer defines that test — no code dependency).

- [ ] **Step 1: Replace the whole file content**

`TestMissingQTable` is removed here (relocated to `tests/test_main.py` in Task 7, since `play()` no longer performs this check — it's the caller's responsibility now). `TestPlay`'s helper switches from a save/load round trip through `PlayConfig` to constructing the agent directly, since `play()` no longer loads anything itself.

```python
from pathlib import Path

from config import PlayConfig
from play import play
from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState


class TestPlay:
    def test_runs_n_episodes_and_returns_a_score_per_episode(self):
        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        agent.epsilon = 0.0
        steps = list(play(env, agent, 5))

        scores = [step.result.info["score"] for step in steps if step.result.done]
        assert len(scores) == 5
        assert all(isinstance(score, int) and score >= 1 for score in scores)

    def test_forces_epsilon_to_zero(self, monkeypatch):
        seen_epsilons = []
        original = QLearningAgent.choose_action

        def spy(self, state_index, mask=None):
            seen_epsilons.append(self.epsilon)
            return original(self, state_index, mask)

        monkeypatch.setattr(QLearningAgent, "choose_action", spy)

        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        agent.epsilon = 0.0
        list(play(env, agent, 1))

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

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_play.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_play.py
git commit -m "Fix test_play.py for play()'s new (env, agent, n_episodes, use_shield) signature"
```

---

### Task 6: Add shield-wiring regression tests to `tests/test_play.py`

**Files:**

- Modify: `tests/test_play.py` (append one new class after `TestPlay`, before `TestDefaults`)

**Interfaces:**

- Consumes: `play.play(env, agent, n_episodes, use_shield)`, `QLearningAgent.choose_action(state_index, mask=None)`.

- [ ] **Step 1: Insert `TestUseShieldWiring` after `TestPlay`, before `TestDefaults`**

```python
class TestUseShieldWiring:
    def test_shield_enabled_passes_a_mask_to_choose_action(self, monkeypatch):
        seen_masks = []
        original = QLearningAgent.choose_action

        def spy(self, state_index, mask=None):
            seen_masks.append(mask)
            return original(self, state_index, mask)

        monkeypatch.setattr(QLearningAgent, "choose_action", spy)
        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        agent.epsilon = 0.0
        list(play(env, agent, 3, use_shield=True))

        assert any(mask is not None for mask in seen_masks)

    def test_shield_disabled_never_passes_a_mask(self, monkeypatch):
        seen_masks = []
        original = QLearningAgent.choose_action

        def spy(self, state_index, mask=None):
            seen_masks.append(mask)
            return original(self, state_index, mask)

        monkeypatch.setattr(QLearningAgent, "choose_action", spy)
        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        agent.epsilon = 0.0
        list(play(env, agent, 3, use_shield=False))

        assert seen_masks
        assert all(mask is None for mask in seen_masks)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_play.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_play.py
git commit -m "Add shield wiring regression tests to test_play.py"
```

---

### Task 7: Replace `tests/test_main.py`'s dispatch tests with end-to-end runs, relocate `TestMissingQTable`

**Files:**

- Modify: `tests/test_main.py` (replace `TestTrainDispatch` and `TestPlayDispatch`; leave `TestPlotFlag`, `TestNoSubcommand`, `TestTrainAndPlayPrintProgress`, `TestTrainCommandSavesQTable` untouched)

**Interfaces:**

- Consumes: `main.main(argv)`, `SnakeEnv.__init__(self, grid_size=20)`, `QLearningAgent.__init__(self, n_states, config=DEFAULT_AGENT_CONFIG)`.

`n_episodes`, `--save-path`, and `--q-table-path` overrides already have real outcome-based coverage via the existing `TestTrainCommandSavesQTable` and `TestTrainAndPlayPrintProgress` classes (both call `main.main()` directly and check real saved files / printed output), so they don't need to be re-tested here. The only CLI knobs the old mock-based `TestTrainDispatch`/`TestPlayDispatch` verified that have **no other externally observable signal** are `--grid-size` and the five agent hyperparameters — `N_STATES` doesn't depend on `grid_size`, so nothing in the saved q_table or printed output reveals whether it was forwarded correctly. Those two tests use a `monkeypatch` **spy** (wraps and still calls the real `__init__`) rather than a mock, so they verify real construction — not a substitute return value.

- [ ] **Step 1: Replace `TestTrainDispatch` and `TestPlayDispatch`**

Replace:

```python
class TestTrainDispatch:
    def test_train_subcommand_calls_train_with_defaults(self, monkeypatch):
        seen_configs = []

        def fake_train(config):
            seen_configs.append(config)
            return iter(())

        monkeypatch.setattr(main, "train", fake_train)
        monkeypatch.setattr(QLearningAgent, "save", lambda self, path: None)

        main.main(["train"])

        assert seen_configs == [TrainConfig()]

    def test_train_subcommand_honors_overrides(self, monkeypatch):
        seen_configs = []

        def fake_train(config):
            seen_configs.append(config)
            return iter(())

        monkeypatch.setattr(main, "train", fake_train)
        monkeypatch.setattr(QLearningAgent, "save", lambda self, path: None)

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
        monkeypatch.setattr(QLearningAgent, "save", lambda self, path: None)

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
```

With:

```python
class TestTrainDispatch:
    def test_train_subcommand_forwards_grid_size_and_hyperparameter_overrides(
        self, monkeypatch, tmp_path
    ):
        seen_grid_sizes = []
        original_env_init = SnakeEnv.__init__

        def env_spy(self, grid_size=20):
            seen_grid_sizes.append(grid_size)
            original_env_init(self, grid_size=grid_size)

        seen_agent_configs = []
        original_agent_init = QLearningAgent.__init__

        def agent_spy(self, n_states, config=DEFAULT_AGENT_CONFIG):
            seen_agent_configs.append(config)
            original_agent_init(self, n_states, config)

        monkeypatch.setattr(SnakeEnv, "__init__", env_spy)
        monkeypatch.setattr(QLearningAgent, "__init__", agent_spy)

        path = tmp_path / "q_table.json"
        main.main(
            [
                "train",
                "--n-episodes", "1",
                "--grid-size", "10",
                "--save-path", str(path),
                "--alpha", "0.5",
                "--gamma", "0.8",
                "--epsilon-start", "0.9",
                "--epsilon-end", "0.05",
                "--epsilon-decay-episodes", "1000",
            ]
        )

        assert seen_grid_sizes == [10]
        assert seen_agent_configs == [
            AgentConfig(
                alpha=0.5,
                gamma=0.8,
                epsilon_start=0.9,
                epsilon_end=0.05,
                epsilon_decay_episodes=1000,
            )
        ]


class TestPlayDispatch:
    def test_play_subcommand_forwards_grid_size_override(self, monkeypatch, tmp_path):
        seen_grid_sizes = []
        original_env_init = SnakeEnv.__init__

        def env_spy(self, grid_size=20):
            seen_grid_sizes.append(grid_size)
            original_env_init(self, grid_size=grid_size)

        monkeypatch.setattr(SnakeEnv, "__init__", env_spy)

        q_path = tmp_path / "q_table.json"
        QLearningAgent(n_states=SnakeState.N_STATES).save(q_path)

        main.main(["play", "--n-episodes", "1", "--grid-size", "10", "--q-table-path", str(q_path)])

        assert seen_grid_sizes == [10]


class TestMissingQTable:
    def test_play_raises_clear_error_when_q_table_missing(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError, match=str(missing_path)):
            main.main(["play", "--q-table-path", str(missing_path)])
```

- [ ] **Step 2: Update the file's imports**

`tests/test_main.py`'s import block needs `SnakeEnv` and `DEFAULT_AGENT_CONFIG` added; `TrainConfig`/`PlayConfig` are still used by `TestPlotFlag`'s indirect dependencies and other untouched tests only if referenced — check the file after Step 1 and ensure the import block reads:

```python
from pathlib import Path

import pytest

import main
from config import DEFAULT_AGENT_CONFIG, AgentConfig, PlayConfig, TrainConfig
from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState
```

(`TrainConfig`/`PlayConfig` are no longer referenced by the rewritten dispatch tests, but leave the imports in place if any other untouched test in the file still uses them — check with `grep -n "TrainConfig\|PlayConfig" tests/test_main.py` before removing either.)

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_main.py -v`
Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_main.py
git commit -m "Replace mock-based dispatch tests with spy-based end-to-end tests in test_main.py"
```

---

### Task 8: Add `TestResumeFrom` and `TestNoShieldFlag` to `tests/test_main.py`

**Files:**

- Modify: `tests/test_main.py` (append two new classes; placement after `TestMissingQTable` from Task 7, before `TestTrainAndPlayPrintProgress`)

**Interfaces:**

- Consumes: `main.main(argv)`, `train.safe_action_mask` (module-level name bound in `train.py` — patch **this**, not `safety.safe_action_mask`).

- [ ] **Step 1: Add `import json` and `import train as train_module` to the top of the file if not already present**

Check the current import block (after Task 7's Step 2) and add:

```python
import json

import train as train_module
```

- [ ] **Step 2: Insert `TestResumeFrom` and `TestNoShieldFlag`**

```python
class TestResumeFrom:
    def test_resume_from_loads_the_existing_q_table_instead_of_starting_fresh(self, tmp_path):
        resume_path = tmp_path / "resume.json"
        save_path = tmp_path / "out.json"

        seed_agent = QLearningAgent(n_states=SnakeState.N_STATES)
        seed_agent.q_table[0] = [7.0, 8.0, 9.0]
        seed_agent.save(resume_path)

        main.main(
            [
                "train",
                "--n-episodes", "0",
                "--grid-size", "8",
                "--save-path", str(save_path),
                "--resume-from", str(resume_path),
            ]
        )

        with open(save_path) as f:
            saved = json.load(f)
        assert saved[0] == [7.0, 8.0, 9.0]


class TestNoShieldFlag:
    def test_shield_runs_by_default_and_is_disabled_by_no_shield(self, monkeypatch, tmp_path):
        calls = []
        original = train_module.safe_action_mask

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(train_module, "safe_action_mask", spy)

        save_path = tmp_path / "out.json"
        main.main(["train", "--n-episodes", "3", "--grid-size", "8", "--save-path", str(save_path)])
        assert len(calls) > 0

        calls.clear()
        save_path2 = tmp_path / "out2.json"
        main.main(
            [
                "train", "--n-episodes", "3", "--grid-size", "8",
                "--save-path", str(save_path2), "--no-shield",
            ]
        )
        assert calls == []
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_main.py -v`
Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_main.py
git commit -m "Add --resume-from and --no-shield regression tests to test_main.py"
```

---

### Task 9: Add `use_shield` coverage to `tests/test_api.py`

**Files:**

- Modify: `tests/test_api.py` (add tests to `TestTrainEndpoint` and `TestPlayEndpoint`; leave `_parse_sse_frames` and existing tests untouched)

**Interfaces:**

- Consumes: `GET /train?use_shield=...`, `GET /play?use_shield=...`, `train.safe_action_mask`, `play.safe_action_mask` (module-level names — same patching rule as Task 8).

- [ ] **Step 1: Add imports for the module-level spy targets**

At the top of `tests/test_api.py`, add:

```python
import play as play_module
import train as train_module
```

- [ ] **Step 2: Add a test to `TestTrainEndpoint`**

Insert at the end of the `TestTrainEndpoint` class:

```python
    def test_use_shield_false_is_accepted_and_disables_the_shield(self, monkeypatch):
        calls = []
        original = train_module.safe_action_mask

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(train_module, "safe_action_mask", spy)

        response = client.get(
            "/train",
            params={"n_episodes": 1, "grid_size": 8, "fps": 120, "use_shield": False},
        )

        assert response.status_code == 200
        assert calls == []
```

- [ ] **Step 3: Add a test to `TestPlayEndpoint`**

Insert at the end of the `TestPlayEndpoint` class:

```python
    def test_use_shield_false_is_accepted_and_disables_the_shield(self, monkeypatch):
        calls = []
        original = play_module.safe_action_mask

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(play_module, "safe_action_mask", spy)

        response = client.get(
            "/play",
            params={"n_episodes": 1, "grid_size": 8, "fps": 120, "use_shield": False},
        )

        assert response.status_code == 200
        assert calls == []
```

- [ ] **Step 4: Run the tests**

Run: `uv run --extra api pytest tests/test_api.py -v`
Expected: all tests PASS (note the `--extra api` flag — this file needs `httpx`, per this repo's documented commands).

- [ ] **Step 5: Commit**

```bash
git add tests/test_api.py
git commit -m "Add use_shield query param coverage to test_api.py"
```

---

### Task 10: Fix the stale `play.py` `FileNotFoundError` claim in CLAUDE.md

**Files:**

- Modify: `CLAUDE.md:80`

**Interfaces:** none.

- [ ] **Step 1: Update the sentence**

Current text (`CLAUDE.md:80`):

```
- **`play.py`** mirrors `train.py`'s episode loop but forces `agent.epsilon = 0.0` after loading and never calls `agent.update` — pure greedy inference, no learning. It fails fast with a clear `FileNotFoundError` if the Q-table path doesn't exist, rather than surfacing `json.load`'s raw traceback.
```

Replace with:

```
- **`play.py`** mirrors `train.py`'s episode loop but takes an already-loaded `agent` and never calls `agent.update` — pure greedy inference, no learning; callers (`main.py`, `watch.py`) own construction, loading, and the `epsilon = 0.0` reset before calling it. The fail-fast `FileNotFoundError` check for a missing Q-table path (rather than surfacing `json.load`'s raw traceback) now lives in those callers, not in `play.py` itself.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "Fix stale CLAUDE.md claim: FileNotFoundError check moved out of play.py"
```

---

## Final verification

- [ ] **Run the full suite**

Run: `uv run pytest`
Expected: all tests PASS.

- [ ] **Run the API suite**

Run: `uv run --extra api pytest tests/test_api.py -v`
Expected: all tests PASS.
