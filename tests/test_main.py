from pathlib import Path

import pytest

import main
from q_agent import QLearningAgent
from snake_state import SnakeState


class TestTrainDispatch:
    def test_train_subcommand_calls_train_with_defaults(self, monkeypatch):
        seen_kwargs = {}

        def fake_train(**kwargs):
            seen_kwargs.update(kwargs)
            return iter(())

        monkeypatch.setattr(main, "train", fake_train)

        main.main(["train"])

        assert seen_kwargs == {
            "n_episodes": 30_000,
            "grid_size": 20,
            "save_path": Path("q_table.json"),
        }

    def test_train_subcommand_honors_overrides(self, monkeypatch):
        seen_kwargs = {}

        def fake_train(**kwargs):
            seen_kwargs.update(kwargs)
            return iter(())

        monkeypatch.setattr(main, "train", fake_train)

        main.main(
            ["train", "--n-episodes", "500", "--grid-size", "10", "--save-path", "out.json"]
        )

        assert seen_kwargs == {
            "n_episodes": 500,
            "grid_size": 10,
            "save_path": Path("out.json"),
        }


class TestPlayDispatch:
    def test_play_subcommand_calls_play_with_defaults(self, monkeypatch):
        seen_kwargs = {}

        def fake_play(**kwargs):
            seen_kwargs.update(kwargs)
            return iter(())

        monkeypatch.setattr(main, "play", fake_play)

        main.main(["play"])

        assert seen_kwargs == {
            "n_episodes": 100,
            "grid_size": 20,
            "q_table_path": Path("q_table.json"),
        }

    def test_play_subcommand_honors_overrides(self, monkeypatch):
        seen_kwargs = {}

        def fake_play(**kwargs):
            seen_kwargs.update(kwargs)
            return iter(())

        monkeypatch.setattr(main, "play", fake_play)

        main.main(
            ["play", "--n-episodes", "5", "--grid-size", "10", "--q-table-path", "other.json"]
        )

        assert seen_kwargs == {
            "n_episodes": 5,
            "grid_size": 10,
            "q_table_path": Path("other.json"),
        }


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
