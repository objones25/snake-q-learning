"""Environment-agnostic safety shield for the snake agent.

Answers one question: "if I take this action from here, do I still have
enough open space afterward to fit my own body?" Built from raw geometry
(a tuple of body cells, a heading, the food cell, and the grid size) — no
dependency on SnakeEnv, Board, or StepResult. Callers that already own an
env (train.py, play.py) read `env.snake` and pass the pieces in; this
module never imports SnakeEnv.
"""

from collections import deque

from snake_types import Action, Direction

Body = tuple[tuple[int, int], ...]


def _resolve_action(
    body: Body,
    direction: Direction,
    action: Action,
    food: tuple[int, int],
    grid_size: int,
) -> Body | None:
    """One-step lookahead. Returns the resulting body, or None on collision.

    Mirrors SnakeEnv.step's turn/move/collision rules exactly, but as a pure
    function over plain tuples — no env, no RNG (food respawn is irrelevant
    to a one-step safety check, so it's left out on purpose).
    """
    if action == Action.RIGHT:
        direction = direction.turn_right()
    elif action == Action.LEFT:
        direction = direction.turn_left()

    new_head = direction.apply(body[-1])
    food_consumed = new_head == food
    out_of_bounds = not (0 <= new_head[0] < grid_size and 0 <= new_head[1] < grid_size)
    tail = body[0]
    occupied = set(body) if food_consumed else set(body) - {tail}

    if out_of_bounds or new_head in occupied:
        return None

    kept = body if food_consumed else body[1:]
    return (*kept, new_head)


def _has_enough_space(body: Body, grid_size: int, needed: int) -> bool:
    """BFS flood fill from the head, short-circuiting once `needed` free cells
    are confirmed reachable. Same reachability rule as counting the full
    region, but skips walking the rest of the board once the answer is
    already decided — the mask only ever needs the True/False threshold, not
    the exact open-cell count.
    """
    if needed <= 0:
        return True
    blocked = set(body)
    head = body[-1]
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque([head])
    while queue:
        x, y = queue.popleft()
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (
                nxt not in visited
                and nxt not in blocked
                and 0 <= nxt[0] < grid_size
                and 0 <= nxt[1] < grid_size
            ):
                visited.add(nxt)
                if len(visited) >= needed:
                    return True
                queue.append(nxt)
    return False


def safe_action_mask(
    body: Body,
    direction: Direction,
    food: tuple[int, int],
    grid_size: int,
) -> tuple[bool, bool, bool]:
    """For each Action: does it avoid both immediate collision and self-trapping?

    An action is unsafe if it collides this step, or if the resulting board
    doesn't have enough BFS-reachable free space to still fit the snake's own
    body (a `reachable < len(body)` pocket means some future cell is
    unreachable without passing through the snake itself). Tuple order
    matches Action's declaration order (STRAIGHT, RIGHT, LEFT), so
    `mask[action]` is valid.
    """
    mask = []
    for action in Action:
        new_body = _resolve_action(body, direction, action, food, grid_size)
        if new_body is None:
            mask.append(False)
        else:
            mask.append(_has_enough_space(new_body, grid_size, len(new_body)))
    return tuple(mask)
