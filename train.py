from collections import deque

from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState


def train(
    n_episodes: int = 10000,
    grid_size: int = 12,
    save_path: str = "q_table.json",
) -> QLearningAgent:
    env = SnakeEnv(grid_size=grid_size)
    agent = QLearningAgent(n_states=SnakeState.N_STATES)
    recent_scores: deque[int] = deque(maxlen=500)

    for episode in range(n_episodes):
        agent.set_epsilon_for_episode(episode)
        state = env.reset()
        result = None
        while result is None or not result.done:
            action = agent.choose_action(state.index)
            result = env.step(action)
            agent.update(
                state.index, action, result.reward,
                result.state.index, result.done, result.truncated,
            )
            state = result.state

        recent_scores.append(result.info["score"])
        if episode % 500 == 0:
            avg_score = sum(recent_scores) / len(recent_scores)
            print(f"episode {episode:6d}  epsilon={agent.epsilon:.3f}  avg_score={avg_score:.2f}")

    agent.save(save_path)
    return agent


if __name__ == "__main__":
    train()
