from episode_step import EpisodeStep
from q_agent import QLearningAgent
from snake_env import Board, StepResult
from snake_state import SnakeState


class TestEpisodeStep:
    def test_round_trips_its_fields(self):
        board = Board(grid_size=8, snake_body=((1, 1),), food=(2, 2))
        state = SnakeState(dng_straight=3, dng_right=3, dng_left=3, food_fwd=0, food_lat=0)
        result = StepResult(
            state=state, reward=0.0, done=False, truncated=False,
            info={"score": 1}, board=board,
        )
        agent = QLearningAgent(n_states=SnakeState.N_STATES)

        step = EpisodeStep(episode=5, result=result, agent=agent)

        assert step.episode == 5
        assert step.result is result
        assert step.agent is agent
