import dataclasses
from pathlib import Path

import pytest

from config import AgentConfig, PlayConfig, RenderConfig, TrainConfig


class TestAgentConfigDefaults:
    def test_defaults_match_current_values(self):
        config = AgentConfig()
        assert config.n_actions == 3
        assert config.alpha == 0.1
        assert config.gamma == 0.9
        assert config.epsilon_start == 1.0
        assert config.epsilon_end == 0.01
        assert config.epsilon_decay_episodes == 5_000

    def test_is_frozen(self):
        config = AgentConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.alpha = 0.5


class TestRenderConfigDefaults:
    def test_defaults_match_current_values(self):
        config = RenderConfig()
        assert config.cell_size == 24
        assert config.fps == 15

    def test_is_frozen(self):
        config = RenderConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.fps = 30


class TestTrainConfigDefaults:
    def test_defaults_match_current_values(self):
        config = TrainConfig()
        assert config.n_episodes == 30_000
        assert config.grid_size == 20
        assert config.save_path == Path("q_table.json")
        assert config.agent == AgentConfig()

    def test_is_frozen(self):
        config = TrainConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.n_episodes = 1


class TestPlayConfigDefaults:
    def test_defaults_match_current_values(self):
        config = PlayConfig()
        assert config.n_episodes == 100
        assert config.grid_size == 20
        assert config.q_table_path == Path("q_table.json")

    def test_is_frozen(self):
        config = PlayConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.n_episodes = 1
