import inspect
import tempfile
from pathlib import Path

import pytest

from play import play
from q_agent import QLearningAgent
from snake_state import SnakeState


class TestMissingQTable:
    def test_raises_clear_error_when_q_table_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "does_not_exist.json"
            with pytest.raises(FileNotFoundError, match=str(missing_path)):
                list(play(n_episodes=1, grid_size=8, q_table_path=missing_path))


class TestPlay:
    def _make_q_table(self, tmpdir):
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        path = Path(tmpdir) / "q_table.json"
        agent.save(path)
        return path

    def test_runs_n_episodes_and_returns_a_score_per_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_q_table(tmpdir)
            steps = list(play(n_episodes=5, grid_size=8, q_table_path=path))

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
            list(play(n_episodes=1, grid_size=8, q_table_path=path))

        assert seen_epsilons
        assert all(epsilon == 0.0 for epsilon in seen_epsilons)


class TestDefaults:
    def test_default_n_episodes_is_100(self):
        assert inspect.signature(play).parameters["n_episodes"].default == 100

    def test_default_grid_size_is_20(self):
        assert inspect.signature(play).parameters["grid_size"].default == 20

    def test_default_q_table_path_is_path_typed(self):
        param = inspect.signature(play).parameters["q_table_path"]
        assert param.annotation is Path
        assert param.default == Path("q_table.json")
