from pathlib import Path

import pytest

import main


class TestTrainDispatch:
    def test_train_subcommand_calls_train_with_defaults(self, monkeypatch):
        seen_kwargs = {}
        monkeypatch.setattr(main, "train", lambda **kwargs: seen_kwargs.update(kwargs))

        main.main(["train"])

        assert seen_kwargs == {
            "n_episodes": 30_000,
            "grid_size": 20,
            "save_path": Path("q_table.json"),
        }

    def test_train_subcommand_honors_overrides(self, monkeypatch):
        seen_kwargs = {}
        monkeypatch.setattr(main, "train", lambda **kwargs: seen_kwargs.update(kwargs))

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
        monkeypatch.setattr(main, "play", lambda **kwargs: seen_kwargs.update(kwargs))

        main.main(["play"])

        assert seen_kwargs == {
            "n_episodes": 100,
            "grid_size": 20,
            "q_table_path": Path("q_table.json"),
        }

    def test_play_subcommand_honors_overrides(self, monkeypatch):
        seen_kwargs = {}
        monkeypatch.setattr(main, "play", lambda **kwargs: seen_kwargs.update(kwargs))

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
