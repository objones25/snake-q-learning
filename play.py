from collections.abc import Iterator

from episode_step import EpisodeStep
from q_agent import QLearningAgent
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
            mask = env.safe_action_mask() if use_shield else None
            action = agent.choose_action(state.index, mask)
            result = env.step(action)
            state = result.state
            yield EpisodeStep(episode=episode, result=result, agent=agent)
