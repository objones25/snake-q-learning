from pathlib import Path

from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState


def play(
    n_episodes: int = 100,
    grid_size: int = 20,
    q_table_path: Path = Path("q_table.json"),
) -> list[int]:
    if not q_table_path.exists():
        raise FileNotFoundError(
            f"No q_table found at {q_table_path} — run `main.py train` first"
        )

    env = SnakeEnv(grid_size=grid_size)
    agent = QLearningAgent(n_states=SnakeState.N_STATES)
    agent.load(q_table_path)
    agent.epsilon = 0.0

    scores = []
    for episode in range(n_episodes):
        state = env.reset()
        result = None
        while result is None or not result.done:
            action = agent.choose_action(state.index)
            result = env.step(action)
            state = result.state

        score = result.info["score"]
        scores.append(score)
        print(f"episode {episode:6d}  score={score}")

    avg_score = sum(scores) / len(scores)
    print(f"avg_score={avg_score:.2f}  top_score={max(scores)}")
    return scores


if __name__ == "__main__":
    play()
