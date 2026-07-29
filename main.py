import argparse
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


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.mode == "train":
        train(n_episodes=args.n_episodes, grid_size=args.grid_size, save_path=args.save_path)
    elif args.mode == "play":
        play(n_episodes=args.n_episodes, grid_size=args.grid_size, q_table_path=args.q_table_path)


if __name__ == "__main__":
    main()
