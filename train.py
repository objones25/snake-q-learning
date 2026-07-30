from collections.abc import Iterator

from episode_step import EpisodeStep
from q_agent import QLearningAgent
from safety import safe_action_mask
from snake_env import SnakeEnv


def _current_mask(env: SnakeEnv) -> tuple[bool, bool, bool]:
    return safe_action_mask(
        tuple(env.snake.body), env.snake.direction, env.food, env.grid_size
    )


def train(
    env: SnakeEnv,
    agent: QLearningAgent,
    n_episodes: int,
    use_shield: bool = True,
) -> Iterator[EpisodeStep]:
    """Run the training loop against an already-constructed env and agent.

    Callers own construction: build a fresh QLearningAgent for a run from
    scratch, or load a saved q_table onto one first to warm-start / continue
    training, or reuse one agent across several envs for a curriculum. This
    function only runs the interaction loop.
    """
    for episode in range(n_episodes):
        agent.set_epsilon_for_episode(episode)
        state = env.reset()
        mask = _current_mask(env) if use_shield else None
        result = None
        while result is None or not result.done:
            action = agent.choose_action(state.index, mask)
            result = env.step(action)
            next_mask = _current_mask(env) if use_shield and not result.done else None
            agent.update(
                state.index,
                action,
                result.reward,
                result.state.index,
                result.done,
                result.truncated,
                next_mask,
            )
            state = result.state
            mask = next_mask
            yield EpisodeStep(episode=episode, result=result, agent=agent)
