from collections.abc import Iterator

from config import DEFAULT_TRAIN_CONFIG, TrainConfig
from episode_step import EpisodeStep
from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState


def train(config: TrainConfig = DEFAULT_TRAIN_CONFIG) -> Iterator[EpisodeStep]:
    env = SnakeEnv(grid_size=config.grid_size)
    agent = QLearningAgent(n_states=SnakeState.N_STATES, config=config.agent)

    for episode in range(config.n_episodes):
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


if __name__ == "__main__":
    for _ in train():
        pass
