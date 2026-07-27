import os
import tempfile

import pytest

from q_agent import QLearningAgent
from snake_types import Action


class TestChooseAction:
    def test_epsilon_zero_always_picks_greedy_action(self):
        agent = QLearningAgent(n_states=5)
        agent.epsilon = 0.0
        agent.q_table[2] = [0.1, 0.9, 0.3]
        assert agent.choose_action(2) == Action.RIGHT

    def test_epsilon_one_always_explores(self):
        agent = QLearningAgent(n_states=5)
        agent.epsilon = 1.0
        agent.q_table[0] = [100.0, 0.0, 0.0]  # STRAIGHT is clearly best...
        seen = {agent.choose_action(0) for _ in range(200)}
        # ...but epsilon=1 should still pick every action at least once
        assert seen == {Action.STRAIGHT, Action.RIGHT, Action.LEFT}

    def test_tie_breaks_to_lowest_index_action(self):
        agent = QLearningAgent(n_states=5)
        agent.epsilon = 0.0
        assert agent.q_table[0] == [0.0, 0.0, 0.0]
        assert agent.choose_action(0) == Action.STRAIGHT


class TestSetEpsilonForEpisode:
    def test_epsilon_starts_at_epsilon_start(self):
        agent = QLearningAgent(
            n_states=5, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_episodes=100
        )
        agent.set_epsilon_for_episode(0)
        assert agent.epsilon == 1.0

    def test_epsilon_reaches_epsilon_end_at_decay_episodes(self):
        agent = QLearningAgent(
            n_states=5, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_episodes=100
        )
        agent.set_epsilon_for_episode(100)
        assert agent.epsilon == pytest.approx(0.01)

    def test_epsilon_holds_at_epsilon_end_past_decay_episodes(self):
        agent = QLearningAgent(
            n_states=5, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_episodes=100
        )
        agent.set_epsilon_for_episode(500)
        assert agent.epsilon == pytest.approx(0.01)

    def test_epsilon_is_linear_at_midpoint(self):
        agent = QLearningAgent(
            n_states=5, epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_episodes=100
        )
        agent.set_epsilon_for_episode(50)
        assert agent.epsilon == pytest.approx(0.5)


class TestUpdate:
    def test_normal_step_bootstraps_with_max_next_q(self):
        agent = QLearningAgent(n_states=5, alpha=0.5, gamma=0.9)
        agent.q_table[1] = [1.0, 2.0, 0.5]  # max = 2.0
        agent.update(
            state_index=0, action=Action.STRAIGHT, reward=1.0,
            next_index=1, done=False, truncated=False,
        )
        # target = 1.0 + 0.9 * 2.0 = 2.8; new_q = 0.0 + 0.5 * (2.8 - 0.0) = 1.4
        assert agent.q_table[0][Action.STRAIGHT] == pytest.approx(1.4)

    def test_real_death_does_not_bootstrap(self):
        agent = QLearningAgent(n_states=5, alpha=0.5, gamma=0.9)
        agent.q_table[1] = [100.0, 100.0, 100.0]  # would blow up the target if bootstrapped
        agent.update(
            state_index=0, action=Action.STRAIGHT, reward=-10.0,
            next_index=1, done=True, truncated=False,
        )
        # target = -10.0 (no bootstrap); new_q = 0.0 + 0.5 * (-10.0 - 0.0) = -5.0
        assert agent.q_table[0][Action.STRAIGHT] == pytest.approx(-5.0)

    def test_truncation_bootstraps_like_a_normal_step(self):
        agent = QLearningAgent(n_states=5, alpha=0.5, gamma=0.9)
        agent.q_table[1] = [1.0, 2.0, 0.5]  # max = 2.0
        agent.update(
            state_index=0, action=Action.STRAIGHT, reward=0.0,
            next_index=1, done=True, truncated=True,
        )
        # target = 0.0 + 0.9 * 2.0 = 1.8; new_q = 0.0 + 0.5 * (1.8 - 0.0) = 0.9
        assert agent.q_table[0][Action.STRAIGHT] == pytest.approx(0.9)


class TestSaveLoad:
    def test_round_trips_q_table_exactly(self):
        agent = QLearningAgent(n_states=3)
        agent.q_table[0] = [1.0, 2.0, 3.0]
        agent.q_table[1] = [4.0, 5.0, 6.0]
        agent.q_table[2] = [7.0, 8.0, 9.0]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "q_table.json")
            agent.save(path)

            loaded_agent = QLearningAgent(n_states=3)
            loaded_agent.load(path)
            assert loaded_agent.q_table == agent.q_table


class TestDefaults:
    def test_default_epsilon_decay_episodes_is_5000(self):
        agent = QLearningAgent(n_states=5)
        assert agent.epsilon_decay_episodes == 5_000
