# Test suite refactor for the flood-fill safety shield

## Context

The `lookahead-floodfill-prototype` branch added a BFS flood-fill safety
shield (`safety.py`, new/untracked) and restructured how `train()`/`play()`
are called:

- `safety.py` exports `safe_action_mask(body, direction, food, grid_size) ->
  (bool, bool, bool)` (order matches `Action`'s declaration order:
  STRAIGHT, RIGHT, LEFT). An action is unsafe if it collides immediately, or
  if the resulting position doesn't have enough BFS-reachable free space to
  fit the snake's own body.
- `QLearningAgent.choose_action(state_index, mask=None)` and
  `QLearningAgent.update(..., next_mask=None)` gained mask parameters:
  selection/bootstrap is restricted to unmasked actions when a mask is
  given and not fully `False`; falls back to the unrestricted set otherwise
  (a boxed-in snake still picks something rather than crashing).
- `train()`/`play()` no longer build their own `env`/`agent` from a
  `TrainConfig`/`PlayConfig` — they now take `(env, agent, n_episodes,
  use_shield=True)`, with callers (`main.py`, `watch.py`, `api.py`) owning
  construction, loading, and (for play) the epsilon=0 / file-existence
  check.
- `main.py` gained `--no-shield` (train and play) and `--resume-from
  <path>` (train only, warm-starts from an existing q_table and resets
  epsilon to `epsilon_start`). `watch.py` gained `--no-shield` (train and
  play, no `--resume-from`). `api.py`'s `/train` and `/play` gained a
  `use_shield` query param.

None of this has test coverage yet. This is a prototype branch (per the
user) and the existing test suite has not been touched — several tests are
now broken by the signature changes, not just missing new coverage.

## Goals

1. Cover the new mechanism itself: `safety.py`'s functions and
   `q_agent.py`'s mask handling.
2. Fix the tests broken by the `train()`/`play()` signature change.
3. Add regression pins proving `use_shield`/`--no-shield`/`--resume-from`
   actually reach the interaction loop, not just that they parse.
4. Add one integration-level soak test exercising the shield wired into a
   real `train()` run, in the spirit of `test_snake_env.py`'s existing
   500-episode soak test.
5. Flag (not necessarily fix in this pass) a stale doc claim in CLAUDE.md.

## What's currently broken (confirmed by reading the diff and running the
files mentally against current signatures)

- `tests/test_train.py`: calls `train(config)` — `TypeError` against the
  new `(env, agent, n_episodes, use_shield)` signature.
- `tests/test_play.py`: calls `play(config)` — same problem.
  `TestMissingQTable` asserts `play(config)` raises `FileNotFoundError` for
  a missing path — that check moved out of `play()` entirely.
- `tests/test_main.py::TestTrainDispatch`/`TestPlayDispatch`: monkeypatch
  `main.train`/`main.play` with single-arg fakes and assert on the config
  object passed — `main.py` no longer calls `train`/`play` with a config
  object, so these mocks receive the wrong arity and the assertions no
  longer describe what `main.py` does.

Not broken, but with no coverage of the new behavior:
`tests/test_q_agent.py` (mask params default to `None`, old tests still
pass but exercise none of the new logic), `safety.py` (zero coverage of
any kind), `tests/test_api.py`/`tests/test_config.py` (still pass, but
don't cover the new `use_shield` param / config fields).

## Design

### A. `tests/test_safety.py` (new)

Matches this repo's convention: `Test*` classes, small helper builders
(no fixtures), heavy parametrize where it fits.

- `TestResolveAction`: turn logic per `Action` (straight/right/left), food
  consumed doesn't shrink the tail, out-of-bounds returns `None`,
  self-collision returns `None` with the tail correctly excluded (mirrors
  `SnakeEnv.step`'s own tail-vacate rule — see CLAUDE.md's note on this
  duplicated logic).
- `TestHasEnoughSpace`: `needed <= 0` is trivially `True`; a boxed region
  with fewer free cells than `needed` is `False`; confirms the
  short-circuit behavior is still correct (doesn't need to walk the whole
  board) via a board where a naive full-flood-fill and the short-circuited
  answer would agree, checking just correctness (not call-count) unless a
  precise short-circuit pin turns out to be cheap to write.
- `TestSafeActionMask`: hand-built boards — wall directly ahead makes that
  action `False`; a one-cell-wide dead-end corridor makes the entering
  action `False` even though the immediate step doesn't collide; a fully
  boxed-in snake returns `(False, False, False)`; tuple order is pinned to
  match `Action`'s declaration order.

### B. `tests/test_q_agent.py` (extend in place)

- `TestChooseAction`: mask restricts both greedy and epsilon-random
  selection to allowed actions; an all-`False` mask falls back to the
  unrestricted set.
- `TestUpdate`: `next_mask` restricts the bootstrap `max` to masked
  actions; an all-`False` `next_mask` falls back to the unrestricted max;
  a variant of the existing real-death test confirms a `next_mask` is
  ignored when `done and not truncated` (bootstrap is skipped regardless
  of mask — pins that the shield can never interfere with this invariant).

### C. `tests/test_train.py` and `tests/test_play.py` (rewrite call sites)

Both switch from building a `TrainConfig`/`PlayConfig` and calling
`train(config)`/`play(config)` to constructing `env`/`agent` directly and
calling `train(env, agent, n_episodes, use_shield)` /
`play(env, agent, n_episodes, use_shield)`.

`tests/test_train.py`:
- Existing `TestTrain` assertions (q_table shape, epsilon decay) port over
  unchanged in spirit.
- `test_death_is_passed_to_update_as_not_truncated`'s spy gains
  `next_mask=None` to match `update`'s new parameter.
- New: with `use_shield=False`, spy on `agent.choose_action`/`agent.update`
  and assert every call's mask arg is `None`. With `use_shield=True`,
  assert at least one call gets a non-`None` mask.
- New **shield soak test**: ~100-200 episodes with a real
  `SnakeEnv`/`QLearningAgent` and `use_shield=True`, asserting per-step
  invariants hold throughout (no exception, snake body/`pos_set`
  consistency) — same style as `test_snake_env.py`'s 500-episode soak
  test, exercising the shield-wired loop instead of the bare env. Lives
  here (not `test_safety.py`) because its subject under test is `train()`'s
  loop, not `safety.py`'s functions directly.

`tests/test_play.py`:
- Same construction switch. `test_forces_epsilon_to_zero`'s spy gains the
  `mask` param.
- Same `use_shield` True/False mask-wiring regression pins as above.
- `TestMissingQTable` is removed from this file — `play()` no longer
  performs this check. Relocated to `tests/test_main.py` (see below).
  `watch.py` has no test file today (consistent with CLAUDE.md's note that
  pygame-dependent code has no automated coverage) and this design doesn't
  add one.

### D. `tests/test_main.py` and `tests/test_api.py`

`tests/test_main.py`:
- `TestTrainDispatch`/`TestPlayDispatch` are replaced with real end-to-end
  runs — call `main.main([...])` for real and assert on outcomes (saved
  q_table exists/has the right shape, printed output), matching this
  file's existing `TestTrainAndPlayPrintProgress`/`TestTrainCommandSavesQTable`
  style. No mocking of `main.train`/`main.play`.
- New `TestResumeFrom`: save a hand-crafted q_table with a known nonzero
  cell, run `main.main(["train", "--resume-from", <path>, "--n-episodes",
  "0", ...])`, assert the saved output still has that value.
- New `TestNoShieldFlag`: monkeypatch `safety.safe_action_mask` with a
  call-counting spy; assert it's called during a normal `train` run and
  never called with `--no-shield`.
- New `TestMissingQTable` (relocated from `test_play.py`):
  `main.main(["play", "--q-table-path", <missing>])` raises
  `FileNotFoundError`.

`tests/test_api.py`:
- Add `use_shield` coverage to `TestTrainEndpoint`/`TestPlayEndpoint`:
  `use_shield=false` is accepted and streams normally (200). Optionally
  the same spy-on-`safe_action_mask` trick to confirm the default
  (`use_shield=true`) actually invokes the shield during a real request —
  keeping with this file's existing real-`TestClient`, no-mocking style.

### E. Documentation

CLAUDE.md's architecture section currently says `play.py` "fails fast with
a clear `FileNotFoundError` if the Q-table path doesn't exist" — stale
since that check now lives in `main.py::_run_play` and
`watch.py::watch_play`. Update this line as part of implementing this
work, so the docs match what the new tests actually pin.

## Out of scope

- `watch.py` gets no new test file — consistent with existing project
  convention (pygame is an optional dependency with no automated
  coverage; CLAUDE.md documents this explicitly for `renderer.py`).
- The duplicated `FileNotFoundError` check between `main.py` and
  `watch.py` is not being refactored into a shared helper — three
  duplicated lines is within this project's stated tolerance for avoiding
  premature abstraction, and `watch.py` isn't gaining a test file that
  would need to exercise it anyway.
- `safety.py`'s unused `_reachable_area` function has already been deleted
  (done outside this spec, prior to writing it).
