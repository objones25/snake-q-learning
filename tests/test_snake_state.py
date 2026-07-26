import dataclasses
from collections import deque

import pytest

from snake import Snake
from snake_state import SnakeState
from snake_types import Direction, Sign


def make_state(dng_straight=False, dng_right=False, dng_left=False,
                food_fwd=Sign.NEG, food_lat=Sign.NEG):
    return SnakeState(
        dng_straight=dng_straight,
        dng_right=dng_right,
        dng_left=dng_left,
        food_fwd=food_fwd,
        food_lat=food_lat,
    )


class TestIndex:
    def test_all_false_min_food_is_zero(self):
        state = make_state()
        assert state.index == 0

    @pytest.mark.parametrize(
        "dng_straight, dng_right, dng_left, expected_danger_bits",
        [
            (False, False, False, 0),
            (False, False, True, 1),
            (False, True, False, 2),
            (False, True, True, 3),
            (True, False, False, 4),
            (True, False, True, 5),
            (True, True, False, 6),
            (True, True, True, 7),
        ],
    )
    def test_danger_bits_scale_index(self, dng_straight, dng_right, dng_left, expected_danger_bits):
        state = make_state(dng_straight=dng_straight, dng_right=dng_right, dng_left=dng_left)
        assert state.index == expected_danger_bits * 9

    @pytest.mark.parametrize(
        "food_fwd, food_lat, expected_food_offset",
        [
            (Sign.NEG, Sign.NEG, 0),
            (Sign.NEG, Sign.ZERO, 1),
            (Sign.NEG, Sign.POS, 2),
            (Sign.ZERO, Sign.NEG, 3),
            (Sign.ZERO, Sign.ZERO, 4),
            (Sign.ZERO, Sign.POS, 5),
            (Sign.POS, Sign.NEG, 6),
            (Sign.POS, Sign.ZERO, 7),
            (Sign.POS, Sign.POS, 8),
        ],
    )
    def test_food_signs_offset_index(self, food_fwd, food_lat, expected_food_offset):
        state = make_state(food_fwd=food_fwd, food_lat=food_lat)
        assert state.index == expected_food_offset

    def test_index_within_bounds_for_all_combinations(self):
        seen = set()
        for dng_straight in (False, True):
            for dng_right in (False, True):
                for dng_left in (False, True):
                    for food_fwd in Sign:
                        for food_lat in Sign:
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
        assert SnakeState.N_STATES == 8 * 9


class TestImmutability:
    def test_is_frozen(self):
        state = make_state()
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.dng_straight = True


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
    def test_dng_straight_true_at_wall(self, head, direction):
        snake = Snake(head, direction)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=10)
        assert state.dng_straight is True

    def test_no_danger_in_open_space(self):
        snake = Snake((5, 5), Direction.RIGHT)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=10)
        assert state.dng_straight is False
        assert state.dng_right is False
        assert state.dng_left is False

    def test_length_one_snake_has_no_self_danger(self):
        snake = Snake((5, 5), Direction.RIGHT)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=10)
        assert state.dng_straight is False
        assert state.dng_right is False
        assert state.dng_left is False

    def test_tail_cell_excluded_from_danger(self):
        # tail=(5,4), head=(5,5); turning to face UP would step onto the
        # tail cell, which vacates on a non-growth move -> not dangerous.
        snake = make_snake([(5, 4), (5, 5)], Direction.RIGHT)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=10)
        assert state.dng_left is False  # RIGHT.turn_left() == UP

    def test_non_tail_body_segment_is_danger(self):
        # tail=(5,3), middle=(5,4), head=(5,5), direction RIGHT.
        # Turning left (UP) steps onto (5,4), a non-tail body segment.
        snake = make_snake([(5, 3), (5, 4), (5, 5)], Direction.RIGHT)
        state = SnakeState.from_world(snake, food=(0, 0), grid_size=10)
        assert state.dng_left is True  # RIGHT.turn_left() == UP -> (5,4)


@pytest.mark.parametrize("direction", list(Direction))
class TestFromWorldFoodSigns:
    def test_food_ahead(self, direction):
        snake = Snake((5, 5), direction)
        food = direction.apply(snake.head)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (Sign.POS, Sign.ZERO)

    def test_food_behind(self, direction):
        snake = Snake((5, 5), direction)
        behind = direction.turn_right().turn_right()
        food = behind.apply(snake.head)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (Sign.NEG, Sign.ZERO)

    def test_food_right(self, direction):
        snake = Snake((5, 5), direction)
        food = direction.turn_right().apply(snake.head)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (Sign.ZERO, Sign.POS)

    def test_food_left(self, direction):
        snake = Snake((5, 5), direction)
        food = direction.turn_left().apply(snake.head)
        state = SnakeState.from_world(snake, food=food, grid_size=20)
        assert (state.food_fwd, state.food_lat) == (Sign.ZERO, Sign.NEG)
