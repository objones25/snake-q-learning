import dataclasses
from collections import deque

import pytest

from snake import Snake
from snake_state import N_DANGER_BUCKETS, SnakeState
from snake_types import Direction


def make_state(dng_straight=0, dng_right=0, dng_left=0, food_fwd=0, food_lat=0):
    return SnakeState(
        dng_straight=dng_straight,
        dng_right=dng_right,
        dng_left=dng_left,
        food_fwd=food_fwd,
        food_lat=food_lat,
    )


class TestIndex:
    def test_min_combination_is_zero(self):
        state = make_state(
            dng_straight=0, dng_right=0, dng_left=0, food_fwd=-2, food_lat=-2
        )
        assert state.index == 0

    @pytest.mark.parametrize(
        "dng_straight, dng_right, dng_left, expected_danger_component",
        [
            (0, 0, 0, 0),
            (0, 0, 1, 1),
            (0, 0, 2, 2),
            (0, 0, 3, 3),
            (0, 1, 0, 4),
            (1, 0, 0, 16),
            (3, 3, 3, 63),
        ],
    )
    def test_danger_component_scales_index_by_25(
        self, dng_straight, dng_right, dng_left, expected_danger_component
    ):
        state = make_state(
            dng_straight=dng_straight,
            dng_right=dng_right,
            dng_left=dng_left,
            food_fwd=-2,
            food_lat=-2,
        )
        assert state.index == expected_danger_component * 25

    @pytest.mark.parametrize(
        "food_fwd, food_lat, expected_food_component",
        [
            (-2, -2, 0),
            (-2, -1, 1),
            (-2, 0, 2),
            (-2, 1, 3),
            (-2, 2, 4),
            (-1, -2, 5),
            (0, -2, 10),
            (0, 0, 12),
            (1, 0, 17),
            (2, 2, 24),
        ],
    )
    def test_food_component_offsets_index(
        self, food_fwd, food_lat, expected_food_component
    ):
        state = make_state(food_fwd=food_fwd, food_lat=food_lat)
        assert state.index == expected_food_component

    def test_index_within_bounds_for_all_combinations(self):
        seen = set()
        for dng_straight in range(N_DANGER_BUCKETS):
            for dng_right in range(N_DANGER_BUCKETS):
                for dng_left in range(N_DANGER_BUCKETS):
                    for food_fwd in range(-2, 3):
                        for food_lat in range(-2, 3):
                            state = make_state(
                                dng_straight=dng_straight,
                                dng_right=dng_right,
                                dng_left=dng_left,
                                food_fwd=food_fwd,
                                food_lat=food_lat,
                            )
                            assert 0 <= state.index < SnakeState.N_STATES
                            seen.add(state.index)
        assert len(seen) == SnakeState.N_STATES

    def test_n_states_matches_total_combinations(self):
        assert SnakeState.N_STATES == 4**3 * 5**2


class TestImmutability:
    def test_is_frozen(self):
        state = make_state()
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.dng_straight = 1  # type: ignore


def make_snake(body: list[tuple[int, int]], direction: Direction) -> Snake:
    snake = Snake(body[0], direction)
    snake.body = deque(body)
    snake.pos_set = set(body)
    return snake


class TestFromWorldDanger:
    @pytest.mark.parametrize(
        "head, direction",
        [
            ((9, 5), Direction.RIGHT),
            ((5, 9), Direction.DOWN),
            ((0, 5), Direction.LEFT),
            ((5, 0), Direction.UP),
        ],
    )
    def test_dng_straight_bucket_0_at_wall(self, head, direction):
        snake = Snake(head, direction)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=10)
        assert state.dng_straight == 0

    def test_non_tail_body_segment_is_bucket_0(self):
        # tail=(5,3), middle=(5,4), head=(5,5), direction RIGHT.
        # Turning left (UP) steps onto (5,4), a non-tail body segment.
        snake = make_snake([(5, 3), (5, 4), (5, 5)], Direction.RIGHT)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=10)
        assert state.dng_left == 0  # RIGHT.turn_left() == UP -> (5,4) adjacent

    def test_open_space_is_bucket_3(self):
        # (10, 10) in a 20-wide grid keeps a 10-cell margin on every side,
        # well beyond the 6-step scan — (5, 5) is only 5 steps from the
        # y=0 wall, which would put dng_left in bucket 2, not 3.
        snake = Snake((10, 10), Direction.RIGHT)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=20)
        assert state.dng_straight == 3
        assert state.dng_right == 3
        assert state.dng_left == 3

    @pytest.mark.parametrize(
        "obstacle_offset, expected_bucket",
        [
            (2, 1),  # obstacle 2 steps away -> ray distance=1 -> bucket 1
            (3, 1),  # obstacle 3 steps away -> ray distance=2 -> bucket 1
            (4, 2),  # obstacle 4 steps away -> ray distance=3 -> bucket 2
            (6, 2),  # obstacle 6 steps away -> ray distance=5 -> bucket 2
            (8, 3),  # obstacle 8 steps away -> beyond MAX_DANGER_SCAN=6 -> bucket 3
        ],
    )
    def test_danger_bucket_scales_with_obstacle_distance(
        self, obstacle_offset, expected_bucket
    ):
        head = (5, 5)
        obstacle = (5 + obstacle_offset, 5)
        tail = (0, 0)  # far away, irrelevant to this ray
        snake = make_snake([tail, obstacle, head], Direction.RIGHT)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=20)
        assert state.dng_straight == expected_bucket

    def test_tail_is_passed_through_when_only_obstacle_in_range(self):
        # tail=(6,5) directly ahead of head=(5,5); nothing else nearby.
        # The tail vacates on this move, so the ray should pass through it
        # and report "no danger within range" (bucket 3), not stop at it.
        snake = make_snake([(6, 5), (5, 5)], Direction.RIGHT)  # tail, head
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=20)
        assert state.dng_straight == 3

    def test_ray_continues_past_tail_to_find_obstacle_beyond(self):
        # tail=(6,5) directly ahead of head, then a non-tail segment further
        # ahead at (8,5). The ray should skip the vacating tail and report
        # the distance to the real obstacle beyond it.
        snake = make_snake([(6, 5), (8, 5), (5, 5)], Direction.RIGHT)  # tail, mid, head
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=20)
        assert state.dng_straight == 1


@pytest.mark.parametrize("direction", list(Direction))
class TestFromWorldFoodBuckets:
    def test_food_near_ahead(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        food = (head[0] + 3 * direction.dx, head[1] + 3 * direction.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (1, 0)

    def test_food_far_ahead(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        food = (head[0] + 4 * direction.dx, head[1] + 4 * direction.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (2, 0)

    def test_food_near_behind(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        behind = direction.turn_right().turn_right()
        food = (head[0] + 3 * behind.dx, head[1] + 3 * behind.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (-1, 0)

    def test_food_far_behind(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        behind = direction.turn_right().turn_right()
        food = (head[0] + 4 * behind.dx, head[1] + 4 * behind.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (-2, 0)

    def test_food_near_right(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        right = direction.turn_right()
        food = (head[0] + 3 * right.dx, head[1] + 3 * right.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (0, 1)

    def test_food_far_right(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        right = direction.turn_right()
        food = (head[0] + 4 * right.dx, head[1] + 4 * right.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (0, 2)

    def test_food_near_left(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        left = direction.turn_left()
        food = (head[0] + 3 * left.dx, head[1] + 3 * left.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (0, -1)

    def test_food_far_left(self, direction):
        snake = Snake((5, 5), direction)
        head = snake.head
        left = direction.turn_left()
        food = (head[0] + 4 * left.dx, head[1] + 4 * left.dy)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (0, -2)

    def test_food_diagonal_offset(self, direction):
        # 5 steps along the forward axis, 2 steps along the right axis:
        # raw fwd component = 5 (far -> magnitude 2), raw lat component = 2
        # (near -> magnitude 1), giving (food_fwd, food_lat) == (2, 1).
        snake = Snake((5, 5), direction)
        head = snake.head
        right = direction.turn_right()
        food = (
            head[0] + 5 * direction.dx + 2 * right.dx,
            head[1] + 5 * direction.dy + 2 * right.dy,
        )
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (2, 1)
