from collections import deque

from snake_types import Direction


class Snake:
    def __init__(self, start_pos: tuple[int, int], direction: int | Direction):
        self.body = deque([start_pos])
        if isinstance(direction, int):
            try:
                direction = Direction(direction)
            except ValueError:
                raise ValueError(
                    f"direction must be one of {[d.value for d in Direction]}, got {direction}"
                )
        self.direction: Direction = direction
        self.pos_set: set = {start_pos}

    @property
    def head(self):
        return self.body[-1]

    @property
    def tail(self):
        return self.body[0]

    @property
    def length(self):
        return len(self.body)

    def turn_right(self) -> None:
        self.direction = self.direction.turn_right()

    def turn_left(self) -> None:
        self.direction = self.direction.turn_left()

    def move(self, food_consumed: bool) -> None:
        new_head = self.direction.apply(self.head)
        if not food_consumed:
            old_tail = self.body.popleft()
            self.pos_set.discard(old_tail)
        self.body.append(new_head)
        self.pos_set.add(new_head)
