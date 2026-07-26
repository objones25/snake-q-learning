from dataclasses import dataclass

from snake_types import Sign


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
