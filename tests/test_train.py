import inspect
import os
import tempfile

from q_agent import QLearningAgent
from snake_state import SnakeState
from train import train


class TestTrain:
    def test_returns_agent_with_correctly_shaped_q_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "q_table.json")
            agent = train(n_episodes=20, grid_size=8, save_path=path)

        assert isinstance(agent, QLearningAgent)
        assert len(agent.q_table) == SnakeState.N_STATES
        assert all(len(row) == 3 for row in agent.q_table)

    def test_epsilon_decreases_from_start_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "q_table.json")
            agent = train(n_episodes=20, grid_size=8, save_path=path)

        assert agent.epsilon < agent.epsilon_start

    def test_saves_q_table_to_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "q_table.json")
            train(n_episodes=20, grid_size=8, save_path=path)
            assert os.path.exists(path)

    def test_death_is_passed_to_update_as_not_truncated(self, monkeypatch):
        seen = []
        original = QLearningAgent.update

        def spy(self, state_index, action, reward, next_index, done, truncated):
            if done:
                seen.append((done, truncated))
            original(self, state_index, action, reward, next_index, done, truncated)

        monkeypatch.setattr(QLearningAgent, "update", spy)
        with tempfile.TemporaryDirectory() as tmpdir:
            train(n_episodes=20, grid_size=8, save_path=os.path.join(tmpdir, "q.json"))

        assert (True, False) in seen


class TestDefaults:
    def test_default_n_episodes_is_200000(self):
        assert inspect.signature(train).parameters["n_episodes"].default == 200_000
