import pytest
from collections import deque

from snake import Snake
from snake_env import SnakeEnv, DEATH_REWARD, FOOD_REWARD, STEP_REWARD
from snake_state import SnakeState
from snake_types import Direction, Action


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


class TestStepTurning:
    def test_straight_keeps_direction(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        original_direction = env.snake.direction
        env.step(Action.STRAIGHT)
        assert env.snake.direction == original_direction

    def test_right_action_turns_right(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        original_direction = env.snake.direction
        env.step(Action.RIGHT)
        assert env.snake.direction == original_direction.turn_right()

    def test_left_action_turns_left(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        original_direction = env.snake.direction
        env.step(Action.LEFT)
        assert env.snake.direction == original_direction.turn_left()


class TestStepCollision:
    def test_wall_collision_ends_episode(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.snake = Snake((11, 5), Direction.RIGHT)
        env.food = (0, 0)
        state, reward, done, info = env.step(Action.STRAIGHT)
        assert done is True
        assert reward == DEATH_REWARD
        assert env.snake.head == (11, 5)  # snake not moved on collision

    def test_self_collision_ends_episode(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        snake = Snake((5, 5), Direction.RIGHT)
        snake.body = deque([(3, 3), (6, 5), (5, 5)])  # tail, non-tail, head
        snake.pos_set = set(snake.body)
        env.snake = snake
        env.food = (0, 0)
        state, reward, done, info = env.step(Action.STRAIGHT)
        assert done is True
        assert reward == DEATH_REWARD

    def test_moving_into_vacated_tail_is_not_collision(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        snake = Snake((5, 5), Direction.RIGHT)
        snake.body = deque([(6, 5), (5, 5)])  # tail, head
        snake.pos_set = set(snake.body)
        env.snake = snake
        env.food = (0, 0)
        state, reward, done, info = env.step(Action.STRAIGHT)
        assert done is False
        assert env.snake.head == (6, 5)
        assert env.snake.length == 2


class TestStepFoodAndReward:
    def test_food_consumption_grows_snake_and_gives_reward(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = env.snake.direction.apply(env.snake.head)
        old_length = env.snake.length
        state, reward, done, info = env.step(Action.STRAIGHT)
        assert reward == FOOD_REWARD
        assert done is False
        assert env.snake.length == old_length + 1
        assert env.steps_since_food == 0
        assert env.food not in env.snake.pos_set
        assert info["score"] == env.snake.length

    def test_non_eating_move_gives_neutral_reward(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        old_length = env.snake.length
        state, reward, done, info = env.step(Action.STRAIGHT)
        assert reward == STEP_REWARD
        assert done is False
        assert env.snake.length == old_length
        assert env.steps_since_food == 1


class TestStepStarvation:
    def test_not_done_at_boundary(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        env.steps_since_food = 100 * env.snake.length - 1
        state, reward, done, info = env.step(Action.STRAIGHT)
        assert done is False

    def test_done_once_over_threshold(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        env.steps_since_food = 100 * env.snake.length
        state, reward, done, info = env.step(Action.STRAIGHT)
        assert done is True
        assert reward == STEP_REWARD
