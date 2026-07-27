from dataclasses import dataclass

from snake import Snake
from snake_types import Direction

MAX_DANGER_SCAN = 6
N_DANGER_BUCKETS = 4
N_FOOD_BUCKETS = 5


def _ray_distance(
    head: tuple[int, int],
    direction: Direction,
    grid_size: int,
    occupied: set[tuple[int, int]],
    max_scan: int = MAX_DANGER_SCAN,
) -> int:
    cell = head
    for distance in range(max_scan):
        cell = direction.apply(cell)
        in_bounds = 0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size
        if not in_bounds or cell in occupied:
            return distance
    return max_scan


def _danger_bucket(distance: int) -> int:
    if distance == 0:
        return 0
    if distance <= 2:
        return 1
    if distance <= 5:
        return 2
    return 3


def _food_bucket(component: int) -> int:
    if component == 0:
        return 0
    magnitude = 1 if abs(component) <= 3 else 2
    return magnitude if component > 0 else -magnitude


@dataclass(frozen=True, slots=True)
class SnakeState:
    dng_straight: int  # 0-3, see _danger_bucket
    dng_right: int
    dng_left: int
    food_fwd: int  # -2..2, see _food_bucket
    food_lat: int

    N_STATES = N_DANGER_BUCKETS ** 3 * N_FOOD_BUCKETS ** 2

    @property
    def index(self) -> int:
        danger_component = (
            self.dng_straight * N_DANGER_BUCKETS + self.dng_right
        ) * N_DANGER_BUCKETS + self.dng_left
        food_component = (self.food_fwd + 2) * N_FOOD_BUCKETS + (self.food_lat + 2)
        return danger_component * (N_FOOD_BUCKETS ** 2) + food_component

    @classmethod
    def from_world(cls, snake: Snake, food: tuple[int, int], grid_size: int) -> "SnakeState":
        head = snake.head
        direction = snake.direction
        occupied = snake.pos_set - {snake.tail}

        dng_straight = _danger_bucket(_ray_distance(head, direction, grid_size, occupied))
        dng_right = _danger_bucket(_ray_distance(head, direction.turn_right(), grid_size, occupied))
        dng_left = _danger_bucket(_ray_distance(head, direction.turn_left(), grid_size, occupied))

        food_vec = (food[0] - head[0], food[1] - head[1])
        fwd_axis = direction.vec
        right_axis = direction.turn_right().vec
        food_fwd = _food_bucket(food_vec[0] * fwd_axis[0] + food_vec[1] * fwd_axis[1])
        food_lat = _food_bucket(food_vec[0] * right_axis[0] + food_vec[1] * right_axis[1])

        return cls(
            dng_straight=dng_straight,
            dng_right=dng_right,
            dng_left=dng_left,
            food_fwd=food_fwd,
            food_lat=food_lat,
        )
