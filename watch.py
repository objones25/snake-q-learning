import argparse
from pathlib import Path

from config import AgentConfig, PlayConfig, RenderConfig, TrainConfig
from play import play
from renderer import PygameRenderer
from train import train


def watch_train(
    train_config: TrainConfig = TrainConfig(),
    render_config: RenderConfig = RenderConfig(),
    render_every: int = 1,
) -> None:
    renderer = PygameRenderer(grid_size=train_config.grid_size, config=render_config)
    try:
        for step in train(train_config):
            if step.episode % render_every != 0:
                continue
            if not renderer.draw(step.result.board, step.episode, step.result.info["score"]):
                break
    finally:
        renderer.close()


def watch_play(
    play_config: PlayConfig = PlayConfig(),
    render_config: RenderConfig = RenderConfig(),
) -> None:
    renderer = PygameRenderer(grid_size=play_config.grid_size, config=render_config)
    try:
        for step in play(play_config):
            if not renderer.draw(step.result.board, step.episode, step.result.info["score"]):
                break
    finally:
        renderer.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="watch.py")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_defaults = TrainConfig()
    agent_defaults = train_defaults.agent
    render_defaults = RenderConfig()

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--n-episodes", type=int, default=train_defaults.n_episodes)
    train_parser.add_argument("--grid-size", type=int, default=train_defaults.grid_size)
    train_parser.add_argument("--save-path", type=Path, default=train_defaults.save_path)
    train_parser.add_argument("--render-every", type=int, default=1)
    train_parser.add_argument("--cell-size", type=int, default=render_defaults.cell_size)
    train_parser.add_argument("--fps", type=int, default=render_defaults.fps)
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
    play_parser.add_argument("--cell-size", type=int, default=render_defaults.cell_size)
    play_parser.add_argument("--fps", type=int, default=render_defaults.fps)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.mode == "train":
        train_config = TrainConfig(
            n_episodes=args.n_episodes,
            grid_size=args.grid_size,
            save_path=args.save_path,
            agent=AgentConfig(
                alpha=args.alpha,
                gamma=args.gamma,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay_episodes=args.epsilon_decay_episodes,
            ),
        )
        render_config = RenderConfig(cell_size=args.cell_size, fps=args.fps)
        watch_train(train_config, render_config, render_every=args.render_every)
    elif args.mode == "play":
        play_config = PlayConfig(
            n_episodes=args.n_episodes,
            grid_size=args.grid_size,
            q_table_path=args.q_table_path,
        )
        render_config = RenderConfig(cell_size=args.cell_size, fps=args.fps)
        watch_play(play_config, render_config)


if __name__ == "__main__":
    main()
