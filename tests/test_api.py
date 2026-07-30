import json

from fastapi.testclient import TestClient

from api import app
from q_agent import QLearningAgent
from snake_env import SnakeEnv

client = TestClient(app)


def _parse_sse_frames(body: str) -> list[dict]:
    frames = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        assert chunk.startswith("data: ")
        frames.append(json.loads(chunk[len("data: ") :]))
    return frames


class TestTrainEndpoint:
    def test_streams_sse_frames_with_expected_keys(self):
        response = client.get(
            "/train", params={"n_episodes": 1, "grid_size": 8, "fps": 120}
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        frames = _parse_sse_frames(response.text)
        assert len(frames) > 0
        for frame in frames:
            assert frame.keys() == {"episode", "board", "score", "reward", "done", "epsilon"}
            assert frame["board"].keys() == {"grid_size", "snake_body", "food"}
            assert frame["epsilon"] is not None

    def test_n_episodes_over_cap_is_rejected(self):
        response = client.get("/train", params={"n_episodes": 201})
        assert response.status_code == 422

    def test_grid_size_out_of_range_is_rejected(self):
        response = client.get("/train", params={"grid_size": 4})
        assert response.status_code == 422

    def test_epsilon_decay_episodes_zero_is_rejected(self):
        response = client.get("/train", params={"epsilon_decay_episodes": 0})
        assert response.status_code == 422

    def test_use_shield_false_is_accepted_and_disables_the_shield(self, monkeypatch):
        calls = []
        original = SnakeEnv.safe_action_mask

        def spy(self):
            calls.append(1)
            return original(self)

        monkeypatch.setattr(SnakeEnv, "safe_action_mask", spy)

        response = client.get(
            "/train",
            params={"n_episodes": 1, "grid_size": 8, "fps": 120, "use_shield": False},
        )

        assert response.status_code == 200
        assert calls == []


class TestPlayEndpoint:
    def test_streams_sse_frames_with_null_epsilon(self):
        response = client.get(
            "/play", params={"n_episodes": 1, "grid_size": 8, "fps": 120}
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        frames = _parse_sse_frames(response.text)
        assert len(frames) > 0
        for frame in frames:
            assert frame.keys() == {"episode", "board", "score", "reward", "done", "epsilon"}
            assert frame["epsilon"] is None

    def test_n_episodes_over_cap_is_rejected(self):
        response = client.get("/play", params={"n_episodes": 101})
        assert response.status_code == 422

    def test_grid_size_out_of_range_is_rejected(self):
        response = client.get("/play", params={"grid_size": 41})
        assert response.status_code == 422

    def test_use_shield_false_is_accepted_and_disables_the_shield(self, monkeypatch):
        calls = []
        original = SnakeEnv.safe_action_mask

        def spy(self):
            calls.append(1)
            return original(self)

        monkeypatch.setattr(SnakeEnv, "safe_action_mask", spy)

        response = client.get(
            "/play",
            params={"n_episodes": 1, "grid_size": 8, "fps": 120, "use_shield": False},
        )

        assert response.status_code == 200
        assert calls == []

    def test_play_forces_epsilon_to_zero_during_the_stream(self, monkeypatch):
        # _stream_play sets agent.epsilon = 0.0 after loading — pin that
        # directly, since play()'s own tests no longer force epsilon
        # themselves (the caller does, and _stream_play is one such caller).
        seen_epsilons = []
        original = QLearningAgent.choose_action

        def spy(self, state_index, mask=None):
            seen_epsilons.append(self.epsilon)
            return original(self, state_index, mask)

        monkeypatch.setattr(QLearningAgent, "choose_action", spy)

        response = client.get(
            "/play", params={"n_episodes": 1, "grid_size": 8, "fps": 120}
        )

        assert response.status_code == 200
        assert seen_epsilons
        assert all(epsilon == 0.0 for epsilon in seen_epsilons)
