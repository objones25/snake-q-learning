from dataclasses import dataclass

from q_agent import QLearningAgent
from snake_env import StepResult


@dataclass(frozen=True, slots=True)
class EpisodeStep:
    episode: int
    result: StepResult
    agent: QLearningAgent
