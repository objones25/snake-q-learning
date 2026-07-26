from dataclasses import dataclass

from snake import Snake
from snake_types import Direction, Sign


@dataclass(frozen=True, slots=True)
class SnakeState:
    dng_straight: bool
    dng_right: bool
    dng_left: bool
    food_fwd: Sign  # POS = ahead of the head, NEG = behind it
    food_lat: Sign  # POS = to the head's right, NEG = to its left

    N_STATES = 72

    @property
    def index(self) -> int:
        d = (self.dng_straight << 2) | (self.dng_right << 1) | self.dng_left
        return d * 9 + (self.food_fwd + 1) * 3 + (self.food_lat + 1)

    @classmethod
    def from_world(cls, snake: Snake, food: tuple[int, int], grid_size: int) -> "SnakeState":
        head = snake.head
        direction = snake.direction
        occupied = snake.pos_set - {snake.tail}

        def is_danger(d: Direction) -> bool:
            cell = d.apply(head)
            in_bounds = 0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size
            return not in_bounds or cell in occupied

        dng_straight = is_danger(direction)
        dng_right = is_danger(direction.turn_right())
        dng_left = is_danger(direction.turn_left())

        food_vec = (food[0] - head[0], food[1] - head[1])
        fwd_axis = direction.vec
        right_axis = direction.turn_right().vec
        food_fwd = Sign.of(food_vec[0] * fwd_axis[0] + food_vec[1] * fwd_axis[1])
        food_lat = Sign.of(food_vec[0] * right_axis[0] + food_vec[1] * right_axis[1])

        return cls(
            dng_straight=dng_straight,
            dng_right=dng_right,
            dng_left=dng_left,
            food_fwd=food_fwd,
            food_lat=food_lat,
        )
