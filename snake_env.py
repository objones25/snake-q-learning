import random

from snake import Snake
from snake_state import SnakeState
from snake_types import Direction


FOOD_REWARD = 10
DEATH_REWARD = -10
STEP_REWARD = 0


class SnakeEnv:
    def __init__(self, grid_size: int = 12):
        self.grid_size = grid_size
        self.snake: Snake | None = None
        self.food: tuple[int, int] | None = None
        self.steps_since_food = 0

    def reset(self) -> SnakeState:
        start_pos = (self.grid_size // 2, self.grid_size // 2)
        direction = random.choice(list(Direction))
        self.snake = Snake(start_pos, direction)
        self.food = self._place_food()
        self.steps_since_food = 0
        return SnakeState.from_world(self.snake, self.food, self.grid_size)

    def _place_food(self) -> tuple[int, int]:
        all_cells = {
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        }
        free_cells = list(all_cells - self.snake.pos_set)
        return random.choice(free_cells)
