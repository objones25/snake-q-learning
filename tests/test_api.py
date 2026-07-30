import json

from fastapi.testclient import TestClient

import play as play_module
import train as train_module

from api import app

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
        original = train_module.safe_action_mask

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(train_module, "safe_action_mask", spy)

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
        original = play_module.safe_action_mask

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(play_module, "safe_action_mask", spy)

        response = client.get(
            "/play",
            params={"n_episodes": 1, "grid_size": 8, "fps": 120, "use_shield": False},
        )

        assert response.status_code == 200
        assert calls == []
