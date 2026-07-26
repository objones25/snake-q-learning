import dataclasses

import pytest

from snake_state import SnakeState
from snake_types import Sign


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
