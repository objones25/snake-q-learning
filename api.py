import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import AgentConfig, PlayConfig, TrainConfig
from episode_step import EpisodeStep
from play import play
from train import train

EXAMPLE_Q_TABLE_PATH = Path(__file__).parent / "example_q_table.json"
MAX_FRAMES_PER_STREAM = 15_000
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])


def _encode(step: EpisodeStep, epsilon: float | None) -> str:
    board = step.result.board
    payload = {
        "episode": step.episode,
        "board": {
            "grid_size": board.grid_size,
            "snake_body": board.snake_body,
            "food": board.food,
        },
        "score": step.result.info["score"],
        "reward": step.result.reward,
        "done": step.result.done,
        "epsilon": epsilon,
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_train(config: TrainConfig, render_every: int, fps: float) -> AsyncIterator[str]:
    delay = 1 / fps
    frames_sent = 0
    for step in train(config):
        if step.episode % render_every != 0:
            continue
        yield _encode(step, epsilon=step.agent.epsilon)
        frames_sent += 1
        if frames_sent >= MAX_FRAMES_PER_STREAM:
            return
        await asyncio.sleep(delay)


async def _stream_play(config: PlayConfig, fps: float) -> AsyncIterator[str]:
    delay = 1 / fps
    frames_sent = 0
    for step in play(config):
        yield _encode(step, epsilon=None)
        frames_sent += 1
        if frames_sent >= MAX_FRAMES_PER_STREAM:
            return
        await asyncio.sleep(delay)


@app.get("/train")
def stream_train(
    n_episodes: int = Query(50, ge=1, le=200),
    grid_size: int = Query(20, ge=8, le=40),
    alpha: float = Query(0.1, ge=0.0, le=1.0),
    gamma: float = Query(0.9, ge=0.0, le=1.0),
    epsilon_start: float = Query(1.0, ge=0.0, le=1.0),
    epsilon_end: float = Query(0.01, ge=0.0, le=1.0),
    epsilon_decay_episodes: int = Query(5_000, ge=1),
    render_every: int = Query(1, ge=1),
    fps: float = Query(30.0, ge=1, le=120),
) -> StreamingResponse:
    config = TrainConfig(
        n_episodes=n_episodes,
        grid_size=grid_size,
        agent=AgentConfig(
            alpha=alpha,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_episodes=epsilon_decay_episodes,
        ),
    )
    return StreamingResponse(
        _stream_train(config, render_every, fps),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.get("/play")
def stream_play(
    n_episodes: int = Query(10, ge=1, le=100),
    grid_size: int = Query(20, ge=8, le=40),
    fps: float = Query(10.0, ge=1, le=120),
) -> StreamingResponse:
    if not EXAMPLE_Q_TABLE_PATH.exists():
        raise HTTPException(status_code=503, detail="Q-table unavailable")
    config = PlayConfig(
        n_episodes=n_episodes, grid_size=grid_size, q_table_path=EXAMPLE_Q_TABLE_PATH
    )
    return StreamingResponse(
        _stream_play(config, fps), media_type="text/event-stream", headers=SSE_HEADERS
    )
