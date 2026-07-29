import argparse
from pathlib import Path

from play import play
from renderer import PygameRenderer
from train import train


def watch_train(
    n_episodes: int = 30_000,
    grid_size: int = 20,
    save_path: Path = Path("q_table.json"),
    render_every: int = 1,
    cell_size: int = 24,
    fps: int = 15,
) -> None:
    renderer = PygameRenderer(grid_size=grid_size, cell_size=cell_size, fps=fps)
    try:
        for step in train(n_episodes=n_episodes, grid_size=grid_size, save_path=save_path):
            if step.episode % render_every != 0:
                continue
            if not renderer.draw(step.result.board, step.episode, step.result.info["score"]):
                break
    finally:
        renderer.close()


def watch_play(
    n_episodes: int = 100,
    grid_size: int = 20,
    q_table_path: Path = Path("q_table.json"),
    cell_size: int = 24,
    fps: int = 15,
) -> None:
    renderer = PygameRenderer(grid_size=grid_size, cell_size=cell_size, fps=fps)
    try:
        for step in play(n_episodes=n_episodes, grid_size=grid_size, q_table_path=q_table_path):
            if not renderer.draw(step.result.board, step.episode, step.result.info["score"]):
                break
    finally:
        renderer.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="watch.py")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--n-episodes", type=int, default=30_000)
    train_parser.add_argument("--grid-size", type=int, default=20)
    train_parser.add_argument("--save-path", type=Path, default=Path("q_table.json"))
    train_parser.add_argument("--render-every", type=int, default=1)
    train_parser.add_argument("--cell-size", type=int, default=24)
    train_parser.add_argument("--fps", type=int, default=15)

    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--n-episodes", type=int, default=100)
    play_parser.add_argument("--grid-size", type=int, default=20)
    play_parser.add_argument("--q-table-path", type=Path, default=Path("q_table.json"))
    play_parser.add_argument("--cell-size", type=int, default=24)
    play_parser.add_argument("--fps", type=int, default=15)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.mode == "train":
        watch_train(
            n_episodes=args.n_episodes,
            grid_size=args.grid_size,
            save_path=args.save_path,
            render_every=args.render_every,
            cell_size=args.cell_size,
            fps=args.fps,
        )
    elif args.mode == "play":
        watch_play(
            n_episodes=args.n_episodes,
            grid_size=args.grid_size,
            q_table_path=args.q_table_path,
            cell_size=args.cell_size,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
