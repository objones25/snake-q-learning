from pathlib import Path

import pytest

from config import AgentConfig, TrainConfig
from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState
from train import train


class TestTrain:
    def test_returns_agent_with_correctly_shaped_q_table(self):
        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        steps = list(train(env, agent, 20))
        result_agent = steps[-1].agent

        assert isinstance(result_agent, QLearningAgent)
        assert len(result_agent.q_table) == SnakeState.N_STATES
        assert all(len(row) == 3 for row in result_agent.q_table)

    def test_epsilon_decreases_from_start_value(self):
        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        steps = list(train(env, agent, 20))

        assert steps[-1].agent.epsilon == pytest.approx(0.996238)

    def test_death_is_passed_to_update_as_not_truncated(self, monkeypatch):
        seen = []
        original = QLearningAgent.update

        def spy(self, state_index, action, reward, next_index, done, truncated, next_mask=None):
            if done:
                seen.append((done, truncated))
            original(self, state_index, action, reward, next_index, done, truncated, next_mask)

        monkeypatch.setattr(QLearningAgent, "update", spy)
        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        list(train(env, agent, 20))

        assert (True, False) in seen


class TestDefaults:
    def test_default_n_episodes_is_30000(self):
        assert TrainConfig().n_episodes == 30_000

    def test_default_grid_size_is_20(self):
        assert TrainConfig().grid_size == 20

    def test_save_path_is_path_typed_with_q_table_json_default(self):
        config = TrainConfig()
        assert isinstance(config.save_path, Path)
        assert config.save_path == Path("q_table.json")

    def test_default_agent_config_matches_agent_config_defaults(self):
        assert TrainConfig().agent == AgentConfig()
