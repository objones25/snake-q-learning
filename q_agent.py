import json
import random

from snake_types import Action


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_episodes: int = 5000,
    ):
        self.q_table: list[list[float]] = [[0.0] * n_actions for _ in range(n_states)]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon = epsilon_start

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

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.q_table, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            self.q_table = json.load(f)
