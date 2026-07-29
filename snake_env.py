import random
from dataclasses import dataclass

from snake import Snake
from snake_state import SnakeState
from snake_types import Action, Direction

FOOD_REWARD = 10
DEATH_REWARD = -10
STEP_REWARD = 0


@dataclass(frozen=True, slots=True)
class StepResult:
    state: SnakeState
    reward: float
    done: bool
    truncated: bool
    info: dict


class SnakeEnv:
    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.snake: Snake
        self.food: tuple[int, int]
        self.steps_since_food = 0
        self._all_cells = {
            (x, y) for x in range(grid_size) for y in range(grid_size)
        }

    def reset(self) -> SnakeState:
        start_pos = (self.grid_size // 2, self.grid_size // 2)
        direction = random.choice(list(Direction))
        self.snake = Snake(start_pos, direction)
        self.food = self._place_food()
        self.steps_since_food = 0
        return SnakeState.from_world(self.snake, self.food, self.grid_size)

    def _place_food(self) -> tuple[int, int]:
        free_cells = list(self._all_cells - self.snake.pos_set)
        return random.choice(free_cells)

    def step(self, action: Action) -> StepResult:
        """Advance the environment by one action.

        `done=True` can mean either termination (wall/self collision) or
        truncation (starvation timeout) — check `truncated` to distinguish
        them, since a training loop typically bootstraps the value estimate
        through truncation but not through real termination.
        """
        if action == Action.RIGHT:
            self.snake.turn_right()
        elif action == Action.LEFT:
            self.snake.turn_left()

        new_head = self.snake.direction.apply(self.snake.head)
        food_consumed = new_head == self.food

        out_of_bounds = not (
            0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size
        )
        occupied = (
            self.snake.pos_set
            if food_consumed
            else self.snake.pos_set - {self.snake.tail}
        )
        collision = out_of_bounds or new_head in occupied

        if collision:
            state = SnakeState.from_world(self.snake, self.food, self.grid_size)
            return StepResult(state, DEATH_REWARD, True, False, {"score": self.snake.length})

        self.snake.move(food_consumed)

        if food_consumed:
            reward = FOOD_REWARD
            self.steps_since_food = 0
            self.food = self._place_food()
        else:
            reward = STEP_REWARD
            self.steps_since_food += 1

        done = self.steps_since_food > 100 * self.snake.length
        state = SnakeState.from_world(self.snake, self.food, self.grid_size)
        return StepResult(state=state, reward=reward, done=done, truncated=done, info={"score": self.snake.length})
