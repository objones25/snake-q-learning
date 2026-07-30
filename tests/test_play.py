from pathlib import Path

from config import PlayConfig
from play import play
from q_agent import QLearningAgent
from snake_env import SnakeEnv
from snake_state import SnakeState


class TestPlay:
    def test_runs_n_episodes_and_returns_a_score_per_episode(self):
        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        agent.epsilon = 0.0
        steps = list(play(env, agent, 5))

        scores = [step.result.info["score"] for step in steps if step.result.done]
        assert len(scores) == 5
        assert all(isinstance(score, int) and score >= 1 for score in scores)

    def test_does_not_modify_the_callers_epsilon(self, monkeypatch):
        # play() no longer forces epsilon to zero itself — that's the
        # caller's job (main.py/watch.py/api.py, before calling play()).
        # This pins that play() never calls anything that would change
        # epsilon out from under the caller.
        seen_epsilons = []
        original = QLearningAgent.choose_action

        def spy(self, state_index, mask=None):
            seen_epsilons.append(self.epsilon)
            return original(self, state_index, mask)

        monkeypatch.setattr(QLearningAgent, "choose_action", spy)

        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        agent.epsilon = 0.0
        list(play(env, agent, 1))

        assert seen_epsilons
        assert all(epsilon == 0.0 for epsilon in seen_epsilons)


class TestUseShieldWiring:
    def test_shield_enabled_passes_a_mask_to_choose_action(self, monkeypatch):
        seen_masks = []
        original = QLearningAgent.choose_action

        def spy(self, state_index, mask=None):
            seen_masks.append(mask)
            return original(self, state_index, mask)

        monkeypatch.setattr(QLearningAgent, "choose_action", spy)
        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        agent.epsilon = 0.0
        list(play(env, agent, 3, use_shield=True))

        assert any(mask is not None for mask in seen_masks)

    def test_shield_disabled_never_passes_a_mask(self, monkeypatch):
        seen_masks = []
        original = QLearningAgent.choose_action

        def spy(self, state_index, mask=None):
            seen_masks.append(mask)
            return original(self, state_index, mask)

        monkeypatch.setattr(QLearningAgent, "choose_action", spy)
        env = SnakeEnv(grid_size=8)
        agent = QLearningAgent(n_states=SnakeState.N_STATES)
        agent.epsilon = 0.0
        list(play(env, agent, 3, use_shield=False))

        assert seen_masks
        assert all(mask is None for mask in seen_masks)


class TestDefaults:
    def test_default_n_episodes_is_100(self):
        assert PlayConfig().n_episodes == 100

    def test_default_grid_size_is_20(self):
        assert PlayConfig().grid_size == 20

    def test_default_q_table_path_is_path_typed(self):
        config = PlayConfig()
        assert isinstance(config.q_table_path, Path)
        assert config.q_table_path == Path("q_table.json")
