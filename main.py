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
from train import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_defaults = DEFAULT_TRAIN_CONFIG
    agent_defaults = train_defaults.agent
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--n-episodes", type=int, default=train_defaults.n_episodes)
    train_parser.add_argument("--grid-size", type=int, default=train_defaults.grid_size)
    train_parser.add_argument("--save-path", type=Path, default=train_defaults.save_path)
    train_parser.add_argument("--alpha", type=float, default=agent_defaults.alpha)
    train_parser.add_argument("--gamma", type=float, default=agent_defaults.gamma)
    train_parser.add_argument("--epsilon-start", type=float, default=agent_defaults.epsilon_start)
    train_parser.add_argument("--epsilon-end", type=float, default=agent_defaults.epsilon_end)
    train_parser.add_argument(
        "--epsilon-decay-episodes", type=int, default=agent_defaults.epsilon_decay_episodes
    )
    train_parser.add_argument("--plot", action="store_true")
    train_parser.add_argument("--plot-path", type=Path, default=None)

    play_defaults = DEFAULT_PLAY_CONFIG
    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--n-episodes", type=int, default=play_defaults.n_episodes)
    play_parser.add_argument("--grid-size", type=int, default=play_defaults.grid_size)
    play_parser.add_argument("--q-table-path", type=Path, default=play_defaults.q_table_path)
    play_parser.add_argument("--plot", action="store_true")
    play_parser.add_argument("--plot-path", type=Path, default=None)

    return parser


def _run_train(config: TrainConfig, plot: bool = False, plot_path: Path | None = None) -> None:
    recent_scores: deque[int] = deque(maxlen=500)
    top_score = 0
    checkpoints: list[tuple[int, float]] = []
    agent = None
    for step in train(config):
        agent = step.agent
        if not step.result.done:
            continue
        recent_scores.append(step.result.info["score"])
        top_score = max(top_score, step.result.info["score"])
        if step.episode % 500 == 0:
            avg_score = sum(recent_scores) / len(recent_scores)
            checkpoints.append((step.episode, avg_score))
            print(
                f"episode {step.episode:6d}  epsilon={step.agent.epsilon:.3f}  "
                f"avg_score={avg_score:.2f}  top_score={top_score}"
            )

    if agent is not None:
        agent.save(config.save_path)

    if plot and checkpoints:
        from plotting import plot_training_progress

        plot_training_progress(checkpoints, save_path=plot_path)


def _run_play(config: PlayConfig, plot: bool = False, plot_path: Path | None = None) -> None:
    scores = []
    for step in play(config):
        if not step.result.done:
            continue
        score = step.result.info["score"]
        scores.append(score)
        print(f"episode {step.episode:6d}  score={score}")

    if scores:  # nothing ran (e.g. --n-episodes 0); avoid dividing by zero
        avg_score = sum(scores) / len(scores)
        print(f"avg_score={avg_score:.2f}  top_score={max(scores)}")

        if plot:
            from plotting import plot_score_distribution

            plot_score_distribution(scores, save_path=plot_path)


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
        )
        _run_train(config, plot=args.plot, plot_path=args.plot_path)
    elif args.mode == "play":
        config = PlayConfig(
            n_episodes=args.n_episodes,
            grid_size=args.grid_size,
            q_table_path=args.q_table_path,
        )
        _run_play(config, plot=args.plot, plot_path=args.plot_path)


if __name__ == "__main__":
    main()
