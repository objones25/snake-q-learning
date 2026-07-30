import argparse
from collections import deque
from pathlib import Path

from config import (
    DEFAULT_PLAY_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    AgentConfig,
    PlayConfig,
    TrainConfig,
)
from play import play
from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState
from train import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_defaults = DEFAULT_TRAIN_CONFIG
    agent_defaults = train_defaults.agent
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--n-episodes", type=int, default=train_defaults.n_episodes
    )
    train_parser.add_argument("--grid-size", type=int, default=train_defaults.grid_size)
    train_parser.add_argument(
        "--save-path", type=Path, default=train_defaults.save_path
    )
    train_parser.add_argument("--alpha", type=float, default=agent_defaults.alpha)
    train_parser.add_argument("--gamma", type=float, default=agent_defaults.gamma)
    train_parser.add_argument(
        "--epsilon-start", type=float, default=agent_defaults.epsilon_start
    )
    train_parser.add_argument(
        "--epsilon-end", type=float, default=agent_defaults.epsilon_end
    )
    train_parser.add_argument(
        "--epsilon-decay-episodes",
        type=int,
        default=agent_defaults.epsilon_decay_episodes,
    )
    train_parser.add_argument("--plot", action="store_true")
    train_parser.add_argument("--plot-path", type=Path, default=None)
    train_parser.add_argument(
        "--no-shield",
        action="store_true",
        help="disable the flood-fill safety shield during training",
    )
    train_parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="warm-start from an existing q_table instead of training from scratch",
    )

    play_defaults = DEFAULT_PLAY_CONFIG
    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--n-episodes", type=int, default=play_defaults.n_episodes)
    play_parser.add_argument("--grid-size", type=int, default=play_defaults.grid_size)
    play_parser.add_argument(
        "--q-table-path", type=Path, default=play_defaults.q_table_path
    )
    play_parser.add_argument("--plot", action="store_true")
    play_parser.add_argument("--plot-path", type=Path, default=None)
    play_parser.add_argument(
        "--no-shield",
        action="store_true",
        help="disable the flood-fill safety shield while playing",
    )

    return parser


def _run_train(config: TrainConfig, resume_from: Path | None = None) -> None:
    env = SnakeEnv(grid_size=config.grid_size)
    agent = QLearningAgent(n_states=SnakeState.N_STATES, config=config.agent)
    if resume_from is not None:
        agent.load(
            resume_from
        )  # warm start: keep config.agent's alpha/gamma/epsilon schedule,
        agent.epsilon = (
            config.agent.epsilon_start
        )  # but pick the Q-table back up where it left off

    recent_scores: deque[int] = deque(maxlen=500)
    top_score = 0
    checkpoints: list[tuple[int, float]] = []
    for step in train(env, agent, config.n_episodes, config.use_shield):
        if not step.result.done:
            continue
        recent_scores.append(step.result.info["score"])
        top_score = max(top_score, step.result.info["score"])
        if step.episode % 500 == 0:
            avg_score = sum(recent_scores) / len(recent_scores)
            checkpoints.append((step.episode, avg_score))
            print(
                f"episode {step.episode:6d}  epsilon={agent.epsilon:.3f}  "
                f"avg_score={avg_score:.2f}  top_score={top_score}"
            )

    agent.save(config.save_path)

    if config.plot and checkpoints:
        from plotting import plot_training_progress

        plot_training_progress(checkpoints, save_path=config.plot_path)


def _run_play(config: PlayConfig) -> None:
    if not config.q_table_path.exists():
        raise FileNotFoundError(
            f"No q_table found at {config.q_table_path} — run `main.py train` first"
        )

    env = SnakeEnv(grid_size=config.grid_size)
    agent = QLearningAgent(n_states=SnakeState.N_STATES)
    agent.load(config.q_table_path)
    agent.epsilon = 0.0

    scores = []
    for step in play(env, agent, config.n_episodes, config.use_shield):
        if not step.result.done:
            continue
        score = step.result.info["score"]
        scores.append(score)
        print(f"episode {step.episode:6d}  score={score}")

    if scores:  # nothing ran (e.g. --n-episodes 0); avoid dividing by zero
        avg_score = sum(scores) / len(scores)
        print(f"avg_score={avg_score:.2f}  top_score={max(scores)}")

        if config.plot:
            from plotting import plot_score_distribution

            plot_score_distribution(scores, save_path=config.plot_path)


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.mode == "train":
        agent_config = AgentConfig(
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_episodes=args.epsilon_decay_episodes,
        )
        config = TrainConfig(
            n_episodes=args.n_episodes,
            grid_size=args.grid_size,
            save_path=args.save_path,
            agent=agent_config,
            use_shield=not args.no_shield,
            plot=args.plot,
            plot_path=args.plot_path,
        )
        _run_train(config, resume_from=args.resume_from)
    elif args.mode == "play":
        config = PlayConfig(
            n_episodes=args.n_episodes,
            grid_size=args.grid_size,
            q_table_path=args.q_table_path,
            use_shield=not args.no_shield,
            plot=args.plot,
            plot_path=args.plot_path,
        )
        _run_play(config)


if __name__ == "__main__":
    main()
