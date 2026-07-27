from collections import deque

import pytest

from snake import Snake
from snake_types import Direction


class TestInit:
    def test_start_pos_and_direction_enum(self):
        snake = Snake((2, 3), Direction.UP)
        assert snake.head == (2, 3)
        assert snake.tail == (2, 3)
        assert snake.length == 1
        assert snake.direction == Direction.UP
        assert snake.pos_set == {(2, 3)}

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0, Direction.RIGHT),
            (1, Direction.DOWN),
            (2, Direction.LEFT),
            (3, Direction.UP),
        ],
    )
    def test_start_direction_as_int(self, value, expected):
        snake = Snake((0, 0), value)
        assert snake.direction == expected

    @pytest.mark.parametrize("value", [-1, 4, 99])
    def test_invalid_int_direction_raises(self, value):
        with pytest.raises(ValueError):
            Snake((0, 0), value)


class TestProperties:
    def test_head_and_tail_differ_after_growth(self):
        snake = Snake((0, 0), Direction.RIGHT)
        snake.move(food_consumed=True)
        assert snake.tail == (0, 0)
        assert snake.head == (1, 0)
        assert snake.length == 2


class TestTurning:
    def test_turn_right(self):
        snake = Snake((0, 0), Direction.RIGHT)
        snake.turn_right()
        assert snake.direction == Direction.DOWN

    def test_turn_left(self):
        snake = Snake((0, 0), Direction.RIGHT)
        snake.turn_left()
        assert snake.direction == Direction.UP


class TestMove:
    def test_move_without_food_keeps_length_constant(self):
        snake = Snake((0, 0), Direction.RIGHT)
        snake.move(food_consumed=False)
        assert snake.length == 1
        assert snake.head == (1, 0)
        assert snake.tail == (1, 0)
        assert snake.pos_set == {(1, 0)}

    def test_move_with_food_grows_snake(self):
        snake = Snake((0, 0), Direction.RIGHT)
        snake.move(food_consumed=True)
        assert snake.length == 2
        assert list(snake.body) == [(0, 0), (1, 0)]
        assert snake.pos_set == {(0, 0), (1, 0)}

    def test_move_direction_applied_to_head(self):
        snake = Snake((5, 5), Direction.UP)
        snake.move(food_consumed=False)
        assert snake.head == (5, 4)

    def test_pos_set_stays_in_sync_over_multiple_moves(self):
        snake = Snake((0, 0), Direction.RIGHT)
        snake.move(food_consumed=True)   # (0,0),(1,0)
        snake.move(food_consumed=True)   # (0,0),(1,0),(2,0)
        snake.move(food_consumed=False)  # drop tail -> (1,0),(2,0),(3,0)
        assert list(snake.body) == [(1, 0), (2, 0), (3, 0)]
        assert snake.pos_set == {(1, 0), (2, 0), (3, 0)}

    def test_move_after_turn_uses_new_direction(self):
        snake = Snake((0, 0), Direction.RIGHT)
        snake.turn_right()  # now DOWN
        snake.move(food_consumed=False)
        assert snake.head == (0, 1)
        assert set(snake.body) == snake.pos_set

    def test_move_into_vacated_tail_keeps_pos_set_in_sync(self):
        # Regression test: head moves into the cell the tail is vacating
        # this same move (a U-shaped body moving "forward"). A naive
        # add-then-discard implementation cancels the two pos_set
        # updates out, silently dropping the new head cell from pos_set.
        snake = Snake((5, 5), Direction.RIGHT)
        snake.body = deque([(6, 5), (5, 5)])  # tail, head
        snake.pos_set = set(snake.body)
        snake.move(food_consumed=False)
        assert list(snake.body) == [(5, 5), (6, 5)]
        assert snake.pos_set == {(5, 5), (6, 5)}
        assert set(snake.body) == snake.pos_set
