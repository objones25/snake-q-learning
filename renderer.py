import pygame

from config import RenderConfig
from snake_env import Board

BG_COLOR = (0, 0, 0)
SNAKE_COLOR = (0, 200, 0)
FOOD_COLOR = (200, 0, 0)


class PygameRenderer:
    def __init__(self, grid_size: int, config: RenderConfig = RenderConfig()):
        pygame.init()
        self.cell_size = config.cell_size
        self.fps = config.fps
        self.screen = pygame.display.set_mode(
            (grid_size * self.cell_size, grid_size * self.cell_size)
        )
        self.clock = pygame.time.Clock()

    def draw(self, board: Board, episode: int, score: int) -> bool:
        """Draws one frame. Returns False if the window was closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(BG_COLOR)
        for cell in board.snake_body:
            pygame.draw.rect(self.screen, SNAKE_COLOR, self._rect(cell))
        pygame.draw.rect(self.screen, FOOD_COLOR, self._rect(board.food))
        pygame.display.set_caption(f"episode {episode}  score {score}")
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def _rect(self, cell: tuple[int, int]) -> pygame.Rect:
        x, y = cell
        return pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)

    def close(self) -> None:
        pygame.quit()
