from pathlib import Path

import pytest

import main
from config import AgentConfig, PlayConfig, TrainConfig
from q_agent import QLearningAgent
from snake_state import SnakeState


class TestTrainDispatch:
    def test_train_subcommand_calls_train_with_defaults(self, monkeypatch):
        seen_configs = []

        def fake_train(config):
            seen_configs.append(config)
            return iter(())

        monkeypatch.setattr(main, "train", fake_train)

        main.main(["train"])

        assert seen_configs == [TrainConfig()]

    def test_train_subcommand_honors_overrides(self, monkeypatch):
        seen_configs = []

        def fake_train(config):
            seen_configs.append(config)
            return iter(())

        monkeypatch.setattr(main, "train", fake_train)

        main.main(
            ["train", "--n-episodes", "500", "--grid-size", "10", "--save-path", "out.json"]
        )

        assert seen_configs == [
            TrainConfig(n_episodes=500, grid_size=10, save_path=Path("out.json"))
        ]

    def test_train_subcommand_honors_agent_hyperparameter_overrides(self, monkeypatch):
        seen_configs = []

        def fake_train(config):
            seen_configs.append(config)
            return iter(())

        monkeypatch.setattr(main, "train", fake_train)

        main.main(
            [
                "train",
                "--alpha", "0.5",
                "--gamma", "0.8",
                "--epsilon-start", "0.9",
                "--epsilon-end", "0.05",
                "--epsilon-decay-episodes", "1000",
            ]
        )

        assert seen_configs == [
            TrainConfig(
                agent=AgentConfig(
                    alpha=0.5,
                    gamma=0.8,
                    epsilon_start=0.9,
                    epsilon_end=0.05,
                    epsilon_decay_episodes=1000,
                )
            )
        ]


class TestPlayDispatch:
    def test_play_subcommand_calls_play_with_defaults(self, monkeypatch):
        seen_configs = []

        def fake_play(config):
            seen_configs.append(config)
            return iter(())

        monkeypatch.setattr(main, "play", fake_play)

        main.main(["play"])

        assert seen_configs == [PlayConfig()]

    def test_play_subcommand_honors_overrides(self, monkeypatch):
        seen_configs = []

        def fake_play(config):
            seen_configs.append(config)
            return iter(())

        monkeypatch.setattr(main, "play", fake_play)

        main.main(
            ["play", "--n-episodes", "5", "--grid-size", "10", "--q-table-path", "other.json"]
        )

        assert seen_configs == [
            PlayConfig(n_episodes=5, grid_size=10, q_table_path=Path("other.json"))
        ]


class TestPlotFlag:
    def test_train_plot_defaults_to_false(self):
        args = main._build_parser().parse_args(["train"])
        assert args.plot is False
        assert args.plot_path is None

    def test_play_plot_defaults_to_false(self):
        args = main._build_parser().parse_args(["play"])
        assert args.plot is False
        assert args.plot_path is None

    def test_train_plot_flags_parse(self):
        args = main._build_parser().parse_args(["train", "--plot", "--plot-path", "out.png"])
        assert args.plot is True
        assert args.plot_path == Path("out.png")

    def test_play_plot_flags_parse(self):
        args = main._build_parser().parse_args(["play", "--plot", "--plot-path", "out.png"])
        assert args.plot is True
        assert args.plot_path == Path("out.png")


class TestNoSubcommand:
    def test_missing_subcommand_exits_with_usage_error(self):
        with pytest.raises(SystemExit):
            main.main([])


class TestTrainAndPlayPrintProgress:
    def test_train_logs_top_score_alongside_avg_score(self, capsys, tmp_path):
        path = tmp_path / "q_table.json"
        main.main(["train", "--n-episodes", "20", "--grid-size", "8", "--save-path", str(path)])

        captured = capsys.readouterr()
        assert "top_score=" in captured.out

    def test_play_prints_per_episode_scores_and_summary(self, capsys, tmp_path):
        q_path = tmp_path / "q_table.json"
        QLearningAgent(n_states=SnakeState.N_STATES).save(q_path)

        main.main(["play", "--n-episodes", "3", "--grid-size", "8", "--q-table-path", str(q_path)])

        captured = capsys.readouterr()
        assert captured.out.count("episode") == 3
        assert "avg_score=" in captured.out
        assert "top_score=" in captured.out


class TestTrainCommandSavesQTable:
    def test_train_command_saves_q_table_to_path(self, tmp_path):
        path = tmp_path / "q_table.json"
        main.main(["train", "--n-episodes", "20", "--grid-size", "8", "--save-path", str(path)])
        assert path.exists()

    def test_train_command_saves_q_table_even_with_zero_episodes(self, tmp_path):
        path = tmp_path / "q_table.json"
        main.main(["train", "--n-episodes", "0", "--grid-size", "8", "--save-path", str(path)])
        assert path.exists()
