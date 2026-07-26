import pytest

from snake_env import SnakeEnv
from snake_state import SnakeState
from snake_types import Direction


class TestReset:
    def test_snake_starts_at_grid_center(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        assert env.snake.head == (6, 6)
        assert env.snake.length == 1

    def test_snake_direction_is_a_valid_direction(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        assert env.snake.direction in list(Direction)

    def test_reset_direction_is_randomized_across_resets(self):
        env = SnakeEnv(grid_size=12)
        directions_seen = set()
        for _ in range(50):
            env.reset()
            directions_seen.add(env.snake.direction)
        assert len(directions_seen) > 1

    def test_food_in_bounds(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        x, y = env.food
        assert 0 <= x < 12
        assert 0 <= y < 12

    def test_food_not_on_snake(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        assert env.food not in env.snake.pos_set

    def test_steps_since_food_reset_to_zero(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        assert env.steps_since_food == 0

    def test_reset_returns_a_snake_state(self):
        env = SnakeEnv(grid_size=12)
        state = env.reset()
        assert isinstance(state, SnakeState)
