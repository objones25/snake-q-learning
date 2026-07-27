import pytest

from snake_types import Action, Direction


class TestAction:
    def test_values(self):
        assert Action.STRAIGHT == 0
        assert Action.RIGHT == 1
        assert Action.LEFT == 2

    def test_is_int_enum_with_three_members(self):
        assert len(list(Action)) == 3


class TestDirection:
    @pytest.mark.parametrize(
        "direction, vec",
        [
            (Direction.RIGHT, (1, 0)),
            (Direction.DOWN, (0, 1)),
            (Direction.LEFT, (-1, 0)),
            (Direction.UP, (0, -1)),
        ],
    )
    def test_vec_dx_dy(self, direction, vec):
        assert direction.vec == vec
        assert direction.dx == vec[0]
        assert direction.dy == vec[1]

    @pytest.mark.parametrize(
        "start, expected",
        [
            (Direction.RIGHT, Direction.DOWN),
            (Direction.DOWN, Direction.LEFT),
            (Direction.LEFT, Direction.UP),
            (Direction.UP, Direction.RIGHT),
        ],
    )
    def test_turn_right(self, start, expected):
        assert start.turn_right() == expected

    @pytest.mark.parametrize(
        "start, expected",
        [
            (Direction.RIGHT, Direction.UP),
            (Direction.UP, Direction.LEFT),
            (Direction.LEFT, Direction.DOWN),
            (Direction.DOWN, Direction.RIGHT),
        ],
    )
    def test_turn_left(self, start, expected):
        assert start.turn_left() == expected

    def test_turn_right_is_turn_left_inverse(self):
        for direction in Direction:
            assert direction.turn_right().turn_left() == direction
            assert direction.turn_left().turn_right() == direction

    def test_four_right_turns_return_to_start(self):
        for direction in Direction:
            result = direction
            for _ in range(4):
                result = result.turn_right()
            assert result == direction

    @pytest.mark.parametrize(
        "direction, p, expected",
        [
            (Direction.RIGHT, (0, 0), (1, 0)),
            (Direction.DOWN, (0, 0), (0, 1)),
            (Direction.LEFT, (0, 0), (-1, 0)),
            (Direction.UP, (0, 0), (0, -1)),
            (Direction.RIGHT, (3, 4), (4, 4)),
        ],
    )
    def test_apply(self, direction, p, expected):
        assert direction.apply(p) == expected
