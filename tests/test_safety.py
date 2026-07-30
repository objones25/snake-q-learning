from safety import _has_enough_space, _resolve_action, safe_action_mask
from snake_types import Action, Direction


class TestResolveAction:
    def test_straight_moves_in_current_direction(self):
        result = _resolve_action(((5, 5),), Direction.RIGHT, Action.STRAIGHT, (0, 0), 10)
        assert result == ((6, 5),)

    def test_right_action_turns_direction_right(self):
        result = _resolve_action(((5, 5),), Direction.UP, Action.RIGHT, (0, 0), 10)
        assert result == ((6, 5),)

    def test_left_action_turns_direction_left(self):
        result = _resolve_action(((5, 5),), Direction.UP, Action.LEFT, (0, 0), 10)
        assert result == ((4, 5),)

    def test_food_consumption_keeps_full_body_and_grows(self):
        result = _resolve_action(((4, 5), (5, 5)), Direction.RIGHT, Action.STRAIGHT, (6, 5), 10)
        assert result == ((4, 5), (5, 5), (6, 5))

    def test_out_of_bounds_returns_none(self):
        result = _resolve_action(((9, 5),), Direction.RIGHT, Action.STRAIGHT, (0, 0), 10)
        assert result is None

    def test_self_collision_returns_none(self):
        # Ring-shaped body; head=(6,5), tail=(5,5). Moving DOWN hits (6,6),
        # a non-tail body segment.
        body = ((5, 5), (5, 6), (6, 6), (6, 5))
        result = _resolve_action(body, Direction.DOWN, Action.STRAIGHT, (0, 0), 10)
        assert result is None

    def test_moving_into_vacated_tail_is_not_a_collision(self):
        body = ((6, 5), (5, 5))  # tail, head
        result = _resolve_action(body, Direction.RIGHT, Action.STRAIGHT, (0, 0), 10)
        assert result == ((5, 5), (6, 5))


class TestHasEnoughSpace:
    def test_needed_zero_or_negative_is_always_true(self):
        # Body fully fills the 1x1 grid (0 free cells) — still True, since
        # needed<=0 short-circuits before any BFS.
        assert _has_enough_space(((0, 0),), 1, 0) is True
        assert _has_enough_space(((0, 0),), 1, -5) is True

    def test_true_when_needed_equals_available_free_cells(self):
        # 5x5 grid = 25 cells; 1 blocked (the head) = 24 free.
        assert _has_enough_space(((2, 2),), 5, 24) is True

    def test_false_when_needed_exceeds_available_free_cells(self):
        assert _has_enough_space(((2, 2),), 5, 25) is False

    def test_true_when_needed_matches_a_small_enclosed_pocket(self):
        # 3x3 grid; body forms a ring around the single center cell (1,1),
        # which is the only free cell reachable from the head.
        ring = ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0))
        assert _has_enough_space(ring, 3, 1) is True

    def test_false_when_needed_exceeds_a_small_enclosed_pocket(self):
        ring = ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0))
        assert _has_enough_space(ring, 3, 2) is False


class TestSafeActionMask:
    def test_wall_ahead_is_unsafe_but_turns_are_safe(self):
        # Head at the right edge of a 5x5 grid, facing further right.
        mask = safe_action_mask(((4, 2),), Direction.RIGHT, (0, 0), 5)
        assert mask == (False, True, True)

    def test_tuple_order_matches_action_declaration_order(self):
        mask = safe_action_mask(((4, 2),), Direction.RIGHT, (0, 0), 5)
        assert mask[Action.STRAIGHT] is False
        assert mask[Action.RIGHT] is True
        assert mask[Action.LEFT] is True

    def test_fully_boxed_snake_returns_all_unsafe(self):
        # None of these bodies need be a physically contiguous snake shape —
        # safe_action_mask only cares about blocked-cell membership. This
        # body/direction combination leaves every one of the 3 candidate
        # moves landing in a pocket too small for the resulting body.
        body = ((0, 1), (2, 1), (1, 2), (1, 0))
        mask = safe_action_mask(body, Direction.DOWN, (2, 2), 3)
        assert mask == (False, False, False)

    def test_moving_into_vacated_tail_is_safe(self):
        mask = safe_action_mask(((6, 5), (5, 5)), Direction.RIGHT, (0, 0), 10)
        assert mask == (True, True, True)
