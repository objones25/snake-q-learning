from collections.abc import Iterator

from config import AgentConfig, PlayConfig
from episode_step import EpisodeStep
from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState


def play(config: PlayConfig = PlayConfig()) -> Iterator[EpisodeStep]:
    if not config.q_table_path.exists():
        raise FileNotFoundError(
            f"No q_table found at {config.q_table_path} — run `main.py train` first"
        )

    env = SnakeEnv(grid_size=config.grid_size)
    agent = QLearningAgent(n_states=SnakeState.N_STATES, config=AgentConfig())
    agent.load(config.q_table_path)
    agent.epsilon = 0.0

    for episode in range(config.n_episodes):
        state = env.reset()
        result = None
        while result is None or not result.done:
            action = agent.choose_action(state.index)
            result = env.step(action)
            state = result.state
            yield EpisodeStep(episode=episode, result=result, agent=agent)


if __name__ == "__main__":
    for _ in play():
        pass
