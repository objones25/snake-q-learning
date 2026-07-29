import argparse
from collections import deque
from pathlib import Path

from play import play
from train import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--n-episodes", type=int, default=30_000)
    train_parser.add_argument("--grid-size", type=int, default=20)
    train_parser.add_argument("--save-path", type=Path, default=Path("q_table.json"))

    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--n-episodes", type=int, default=100)
    play_parser.add_argument("--grid-size", type=int, default=20)
    play_parser.add_argument("--q-table-path", type=Path, default=Path("q_table.json"))

    return parser


def _run_train(n_episodes: int, grid_size: int, save_path: Path) -> None:
    recent_scores: deque[int] = deque(maxlen=500)
    top_score = 0
    for step in train(n_episodes=n_episodes, grid_size=grid_size, save_path=save_path):
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


def _run_play(n_episodes: int, grid_size: int, q_table_path: Path) -> None:
    scores = []
    for step in play(n_episodes=n_episodes, grid_size=grid_size, q_table_path=q_table_path):
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
        _run_train(n_episodes=args.n_episodes, grid_size=args.grid_size, save_path=args.save_path)
    elif args.mode == "play":
        _run_play(n_episodes=args.n_episodes, grid_size=args.grid_size, q_table_path=args.q_table_path)


if __name__ == "__main__":
    main()
