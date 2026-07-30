from collections.abc import Iterator

from episode_step import EpisodeStep
from q_agent import QLearningAgent
from safety import safe_action_mask
from snake_env import SnakeEnv


def play(
    env: SnakeEnv,
    agent: QLearningAgent,
    n_episodes: int,
    use_shield: bool = True,
) -> Iterator[EpisodeStep]:
    """Run the play loop against an already-constructed env and agent.

    Callers own construction — loading the q_table, setting epsilon=0, and
    checking the file exists all happen before this is called.
    """
    for episode in range(n_episodes):
        state = env.reset()
        result = None
        while result is None or not result.done:
            mask = None
            if use_shield:
                mask = safe_action_mask(
                    tuple(env.snake.body), env.snake.direction, env.food, env.grid_size
                )
            action = agent.choose_action(state.index, mask)
            result = env.step(action)
            state = result.state
            yield EpisodeStep(episode=episode, result=result, agent=agent)
