import argparse
from collections import deque
from pathlib import Path

from config import AgentConfig, PlayConfig, TrainConfig
from play import play
from train import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_defaults = TrainConfig()
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

    play_defaults = PlayConfig()
    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--n-episodes", type=int, default=play_defaults.n_episodes)
    play_parser.add_argument("--grid-size", type=int, default=play_defaults.grid_size)
    play_parser.add_argument("--q-table-path", type=Path, default=play_defaults.q_table_path)

    return parser


def _run_train(config: TrainConfig) -> None:
    recent_scores: deque[int] = deque(maxlen=500)
    top_score = 0
    for step in train(config):
        if not step.result.done:
            continue
        recent_scores.append(step.result.info["score"])
        top_score = max(top_score, step.result.info["score"])
        if step.episode % 500 == 0:
            avg_score = sum(recent_scores) / len(recent_scores)
            print(
                f"episode {step.episode:6d}  epsilon={step.agent.epsilon:.3f}  "
                f"avg_score={avg_score:.2f}  top_score={top_score}"
            )


def _run_play(config: PlayConfig) -> None:
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
        _run_train(config)
    elif args.mode == "play":
        config = PlayConfig(
            n_episodes=args.n_episodes,
            grid_size=args.grid_size,
            q_table_path=args.q_table_path,
        )
        _run_play(config)


if __name__ == "__main__":
    main()
