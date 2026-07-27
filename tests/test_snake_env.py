import random
from collections import deque

import pytest

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
        result = env.step(Action.STRAIGHT)
        assert result.done is True
        assert result.truncated is False
        assert result.reward == DEATH_REWARD
        assert env.snake.head == (11, 5)  # snake not moved on collision

    def test_self_collision_ends_episode(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        snake = Snake((5, 5), Direction.RIGHT)
        snake.body = deque([(3, 3), (6, 5), (5, 5)])  # tail, non-tail, head
        snake.pos_set = set(snake.body)
        env.snake = snake
        env.food = (0, 0)
        result = env.step(Action.STRAIGHT)
        assert result.done is True
        assert result.reward == DEATH_REWARD

    def test_moving_into_vacated_tail_is_not_collision(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        snake = Snake((5, 5), Direction.RIGHT)
        snake.body = deque([(6, 5), (5, 5)])  # tail, head
        snake.pos_set = set(snake.body)
        env.snake = snake
        env.food = (0, 0)
        result = env.step(Action.STRAIGHT)
        assert result.done is False
        assert env.snake.head == (6, 5)
        assert env.snake.length == 2
        assert set(env.snake.body) == env.snake.pos_set


class TestStepFoodAndReward:
    def test_food_consumption_grows_snake_and_gives_reward(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = env.snake.direction.apply(env.snake.head)
        old_length = env.snake.length
        result = env.step(Action.STRAIGHT)
        assert result.reward == FOOD_REWARD
        assert result.done is False
        assert env.snake.length == old_length + 1
        assert env.steps_since_food == 0
        assert env.food not in env.snake.pos_set
        assert result.info["score"] == env.snake.length

    def test_non_eating_move_gives_neutral_reward(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        old_length = env.snake.length
        result = env.step(Action.STRAIGHT)
        assert result.reward == STEP_REWARD
        assert result.done is False
        assert env.snake.length == old_length
        assert env.steps_since_food == 1


class TestStepStarvation:
    def test_not_done_at_boundary(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        env.steps_since_food = 100 * env.snake.length - 1
        result = env.step(Action.STRAIGHT)
        assert result.done is False

    def test_done_once_over_threshold(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        env.steps_since_food = 100 * env.snake.length
        result = env.step(Action.STRAIGHT)
        assert result.done is True
        assert result.reward == STEP_REWARD

    def test_starvation_timeout_marks_result_as_truncated(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        env.steps_since_food = 100 * env.snake.length
        result = env.step(Action.STRAIGHT)
        assert result.done is True
        assert result.truncated is True


class TestStepTruncatedFlag:
    def test_death_path_is_not_truncated(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.snake = Snake((11, 5), Direction.RIGHT)
        env.food = (0, 0)
        result = env.step(Action.STRAIGHT)
        assert result.done is True
        assert result.truncated is False

    def test_food_path_is_not_truncated(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = env.snake.direction.apply(env.snake.head)
        result = env.step(Action.STRAIGHT)
        assert result.reward == FOOD_REWARD
        assert result.truncated is False

    def test_non_eating_move_is_not_truncated(self):
        env = SnakeEnv(grid_size=12)
        env.reset()
        env.food = (0, 0)
        result = env.step(Action.STRAIGHT)
        assert result.reward == STEP_REWARD
        assert result.truncated is False


class TestLifecycle:
    def test_random_policy_episodes_hold_invariants_every_step(self):
        # Realistic reset() -> step() -> ... -> done loop under a random
        # policy. This is what would have caught the Snake.move() pos_set
        # corruption bug: single hand-assembled steps never exercised the
        # "head moves into the cell the tail is vacating this same move"
        # case, so the desync went unnoticed by 105 green tests.
        #
        # SnakeEnv itself draws its randomness (start direction, food
        # placement) from the shared `random` module rather than an
        # injected Random instance, so the module-level seed is what we
        # need to pin for a reproducible, deterministic test - a
        # locally-scoped random.Random(...) would only control our own
        # action choices and leave the env's internal randomness (and
        # therefore whether this test actually exercises the bug)
        # dependent on whatever happened to run before it.
        grid_size = 8
        num_episodes = 500
        max_steps_per_episode = 1000
        random.seed(1)

        env = SnakeEnv(grid_size=grid_size)

        for _ in range(num_episodes):
            env.reset()
            reached_done = False

            for _ in range(max_steps_per_episode):
                action = random.choice(list(Action))
                result = env.step(action)

                body = env.snake.body
                assert set(body) == env.snake.pos_set
                assert len(body) == len(set(body))
                for x, y in body:
                    assert 0 <= x < grid_size
                    assert 0 <= y < grid_size

                fx, fy = env.food
                assert 0 <= fx < grid_size
                assert 0 <= fy < grid_size
                assert env.food not in env.snake.pos_set

                assert 0 <= result.state.index < SnakeState.N_STATES
                assert result.reward in (FOOD_REWARD, DEATH_REWARD, STEP_REWARD)
                assert result.info["score"] == env.snake.length

                if result.done:
                    reached_done = True
                    break

            assert reached_done, (
                f"episode did not reach done within {max_steps_per_episode} steps"
            )
