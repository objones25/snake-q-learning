import tempfile
from pathlib import Path

import pytest

from config import PlayConfig
from play import play
from q_agent import QLearningAgent
from snake_state import SnakeState


class TestMissingQTable:
    def test_raises_clear_error_when_q_table_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "does_not_exist.json"
            config = PlayConfig(n_episodes=1, grid_size=8, q_table_path=missing_path)
            with pytest.raises(FileNotFoundError, match=str(missing_path)):
                list(play(config))


class TestPlay:
    def _make_q_table(self, tmpdir):
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        path = Path(tmpdir) / "q_table.json"
        agent.save(path)
        return path

    def test_runs_n_episodes_and_returns_a_score_per_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_q_table(tmpdir)
            config = PlayConfig(n_episodes=5, grid_size=8, q_table_path=path)
            steps = list(play(config))

        scores = [step.result.info["score"] for step in steps if step.result.done]
        assert len(scores) == 5
        assert all(isinstance(score, int) and score >= 1 for score in scores)

    def test_forces_epsilon_to_zero(self, monkeypatch):
        seen_epsilons = []
        original = QLearningAgent.choose_action

        def spy(self, state_index):
            seen_epsilons.append(self.epsilon)
            return original(self, state_index)

        monkeypatch.setattr(QLearningAgent, "choose_action", spy)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_q_table(tmpdir)
            config = PlayConfig(n_episodes=1, grid_size=8, q_table_path=path)
            list(play(config))

        assert seen_epsilons
        assert all(epsilon == 0.0 for epsilon in seen_epsilons)


class TestDefaults:
    def test_default_n_episodes_is_100(self):
        assert PlayConfig().n_episodes == 100

    def test_default_grid_size_is_20(self):
        assert PlayConfig().grid_size == 20

    def test_default_q_table_path_is_path_typed(self):
        config = PlayConfig()
        assert isinstance(config.q_table_path, Path)
        assert config.q_table_path == Path("q_table.json")
