import json
from pathlib import Path

import pytest

import main
from config import DEFAULT_AGENT_CONFIG, AgentConfig
from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState


class TestTrainDispatch:
    def test_train_subcommand_forwards_grid_size_and_hyperparameter_overrides(
        self, monkeypatch, tmp_path
    ):
        seen_grid_sizes = []
        original_env_init = SnakeEnv.__init__

        def env_spy(self, grid_size=20):
            seen_grid_sizes.append(grid_size)
            original_env_init(self, grid_size=grid_size)

        seen_agent_configs = []
        original_agent_init = QLearningAgent.__init__

        def agent_spy(self, n_states, config=DEFAULT_AGENT_CONFIG):
            seen_agent_configs.append(config)
            original_agent_init(self, n_states, config)

        monkeypatch.setattr(SnakeEnv, "__init__", env_spy)
        monkeypatch.setattr(QLearningAgent, "__init__", agent_spy)

        path = tmp_path / "q_table.json"
        main.main(
            [
                "train",
                "--n-episodes", "1",
                "--grid-size", "10",
                "--save-path", str(path),
                "--alpha", "0.5",
                "--gamma", "0.8",
                "--epsilon-start", "0.9",
                "--epsilon-end", "0.05",
                "--epsilon-decay-episodes", "1000",
            ]
        )

        assert seen_grid_sizes == [10]
        assert seen_agent_configs == [
            AgentConfig(
                alpha=0.5,
                gamma=0.8,
                epsilon_start=0.9,
                epsilon_end=0.05,
                epsilon_decay_episodes=1000,
            )
        ]


class TestPlayDispatch:
    def test_play_subcommand_forwards_grid_size_override(self, monkeypatch, tmp_path):
        seen_grid_sizes = []
        original_env_init = SnakeEnv.__init__

        def env_spy(self, grid_size=20):
            seen_grid_sizes.append(grid_size)
            original_env_init(self, grid_size=grid_size)

        monkeypatch.setattr(SnakeEnv, "__init__", env_spy)

        q_path = tmp_path / "q_table.json"
        QLearningAgent(n_states=SnakeState.N_STATES).save(q_path)

        main.main(["play", "--n-episodes", "1", "--grid-size", "10", "--q-table-path", str(q_path)])

        assert seen_grid_sizes == [10]

    def test_play_forces_epsilon_to_zero_before_running(self, monkeypatch, tmp_path):
        # _run_play sets agent.epsilon = 0.0 after loading — this is the only
        # place that enforces "play is greedy" for the main.py CLI entry
        # point, so pin it directly rather than relying on play()'s own
        # tests (which no longer force epsilon themselves; the caller does).
        seen_epsilons = []
        original = QLearningAgent.choose_action

        def spy(self, state_index, mask=None):
            seen_epsilons.append(self.epsilon)
            return original(self, state_index, mask)

        monkeypatch.setattr(QLearningAgent, "choose_action", spy)

        q_path = tmp_path / "q_table.json"
        QLearningAgent(n_states=SnakeState.N_STATES).save(q_path)

        main.main(["play", "--n-episodes", "3", "--grid-size", "8", "--q-table-path", str(q_path)])

        assert seen_epsilons
        assert all(epsilon == 0.0 for epsilon in seen_epsilons)


class TestMissingQTable:
    def test_play_raises_clear_error_when_q_table_missing(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError, match=str(missing_path)):
            main.main(["play", "--q-table-path", str(missing_path)])


class TestResumeFrom:
    def test_resume_from_loads_the_existing_q_table_instead_of_starting_fresh(self, tmp_path):
        resume_path = tmp_path / "resume.json"
        save_path = tmp_path / "out.json"

        seed_agent = QLearningAgent(n_states=SnakeState.N_STATES)
        seed_agent.q_table[0] = [7.0, 8.0, 9.0]
        seed_agent.save(resume_path)

        main.main(
            [
                "train",
                "--n-episodes", "0",
                "--grid-size", "8",
                "--save-path", str(save_path),
                "--resume-from", str(resume_path),
            ]
        )

        with open(save_path) as f:
            saved = json.load(f)
        assert saved[0] == [7.0, 8.0, 9.0]


class TestNoShieldFlag:
    def test_shield_runs_by_default_and_is_disabled_by_no_shield(self, monkeypatch, tmp_path):
        calls = []
        original = SnakeEnv.safe_action_mask

        def spy(self):
            calls.append(1)
            return original(self)

        monkeypatch.setattr(SnakeEnv, "safe_action_mask", spy)

        save_path = tmp_path / "out.json"
        main.main(["train", "--n-episodes", "3", "--grid-size", "8", "--save-path", str(save_path)])
        assert len(calls) > 0

        calls.clear()
        save_path2 = tmp_path / "out2.json"
        main.main(
            [
                "train", "--n-episodes", "3", "--grid-size", "8",
                "--save-path", str(save_path2), "--no-shield",
            ]
        )
        assert calls == []


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
