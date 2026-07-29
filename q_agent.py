import json
import random
from pathlib import Path

from config import AgentConfig
from snake_types import Action


class QLearningAgent:
    def __init__(self, n_states: int, config: AgentConfig = AgentConfig()):
        self.q_table: list[list[float]] = [[0.0] * config.n_actions for _ in range(n_states)]
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay_episodes = config.epsilon_decay_episodes
        self.epsilon = config.epsilon_start

    def set_epsilon_for_episode(self, episode: int) -> None:
        fraction = min(episode / self.epsilon_decay_episodes, 1.0)
        self.epsilon = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_end)

    def choose_action(self, state_index: int) -> Action:
        if random.random() < self.epsilon:
            return random.choice(list(Action))
        q_values = self.q_table[state_index]
        best_action = max(range(len(q_values)), key=lambda a: q_values[a])
        return Action(best_action)

    def update(
        self,
        state_index: int,
        action: Action,
        reward: float,
        next_index: int,
        done: bool,
        truncated: bool,
    ) -> None:
        current = self.q_table[state_index][action]
        if done and not truncated:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_index])
        self.q_table[state_index][action] += self.alpha * (target - current)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.q_table, f)

    def load(self, path: Path) -> None:
        with open(path) as f:
            self.q_table = json.load(f)
