from enum import Enum, IntEnum


class Sign(IntEnum):
    NEG = -1
    ZERO = 0
    POS = 1

    @staticmethod
    def of(x: int) -> "Sign":
        return Sign(0 if x == 0 else (1 if x > 0 else -1))


class Action(IntEnum):
    STRAIGHT = 0
    RIGHT = 1
    LEFT = 2


class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    @property
    def vec(self) -> tuple[int, int]:
        return _VECS[self.value]

    @property
    def dx(self) -> int:
        return _VECS[self.value][0]

    @property
    def dy(self) -> int:
        return _VECS[self.value][1]

    def turn_right(self) -> "Direction":
        return Direction((self.value + 1) % 4)

    def turn_left(self) -> "Direction":
        return Direction((self.value - 1) % 4)

    def apply(self, p: tuple[int, int]) -> tuple[int, int]:
        return (p[0] + self.dx, p[1] + self.dy)


_VECS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))
