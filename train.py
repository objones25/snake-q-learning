from collections.abc import Iterator
from pathlib import Path

from episode_step import EpisodeStep
from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState


def train(
    n_episodes: int = 30_000,
    grid_size: int = 20,
    save_path: Path = Path("q_table.json"),
) -> Iterator[EpisodeStep]:
    env = SnakeEnv(grid_size=grid_size)
    agent = QLearningAgent(n_states=SnakeState.N_STATES)

    for episode in range(n_episodes):
        agent.set_epsilon_for_episode(episode)
        state = env.reset()
        result = None
        while result is None or not result.done:
            action = agent.choose_action(state.index)
            result = env.step(action)
            agent.update(
                state.index,
                action,
                result.reward,
                result.state.index,
                result.done,
                result.truncated,
            )
            state = result.state
            yield EpisodeStep(episode=episode, result=result, agent=agent)

    agent.save(save_path)


if __name__ == "__main__":
    for _ in train():
        pass
