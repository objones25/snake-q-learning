import json
import random
from pathlib import Path

from config import DEFAULT_AGENT_CONFIG, AgentConfig
from snake_types import Action


class QLearningAgent:
    def __init__(self, n_states: int, config: AgentConfig = DEFAULT_AGENT_CONFIG):
        self.q_table: list[list[float]] = [
            [0.0] * config.n_actions for _ in range(n_states)
        ]
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay_episodes = config.epsilon_decay_episodes
        self.epsilon = config.epsilon_start

    def set_epsilon_for_episode(self, episode: int) -> None:
        fraction = min(episode / self.epsilon_decay_episodes, 1.0)
        self.epsilon = self.epsilon_start - fraction * (
            self.epsilon_start - self.epsilon_end
        )

    def choose_action(
        self, state_index: int, mask: tuple[bool, ...] | None = None
    ) -> Action:
        """Pick an action, optionally restricted to a safety mask.

        `mask` is a per-action boolean tuple (see `safety.safe_action_mask`):
        True means the action is allowed. If every action is masked out
        (fully boxed in, no escape), the mask is ignored for this call so the
        agent still picks something rather than crashing.
        """
        candidates = (
            list(Action)
            if mask is None or not any(mask)
            else [a for a in Action if mask[a]]
        )
        if random.random() < self.epsilon:
            return random.choice(candidates)
        q_values = self.q_table[state_index]
        best_action = max(candidates, key=lambda a: q_values[a])
        return Action(best_action)

    def update(
        self,
        state_index: int,
        action: Action,
        reward: float,
        next_index: int,
        done: bool,
        truncated: bool,
        next_mask: tuple[bool, ...] | None = None,
    ) -> None:
        """Q-learning update. `next_mask`, if given, restricts the bootstrap
        max to actions the shield would actually allow from the next state —
        keeping the learned values consistent with a policy that never takes
        masked actions. Falls back to the full max if nothing is masked-safe.
        """
        current = self.q_table[state_index][action]
        if done and not truncated:
            target = reward
        else:
            next_q = self.q_table[next_index]
            if next_mask is not None and any(next_mask):
                bootstrap = max(q for a, q in enumerate(next_q) if next_mask[a])
            else:
                bootstrap = max(next_q)
            target = reward + self.gamma * bootstrap
        self.q_table[state_index][action] += self.alpha * (target - current)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.q_table, f)

    def load(self, path: Path) -> None:
        with open(path) as f:
            self.q_table = json.load(f)
