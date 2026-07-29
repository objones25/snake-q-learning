from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    n_actions: int = 3
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 5_000


@dataclass(frozen=True)
class RenderConfig:
    cell_size: int = 24
    fps: int = 15


@dataclass(frozen=True)
class TrainConfig:
    n_episodes: int = 30_000
    grid_size: int = 20
    save_path: Path = Path("q_table.json")
    agent: AgentConfig = field(default_factory=AgentConfig)


@dataclass(frozen=True)
class PlayConfig:
    n_episodes: int = 100
    grid_size: int = 20
    q_table_path: Path = Path("q_table.json")
