import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import AgentConfig, PlayConfig, TrainConfig
from episode_step import EpisodeStep
from play import play
from train import train

EXAMPLE_Q_TABLE_PATH = Path("example_q_table.json")

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


async def _stream_train(config: TrainConfig, render_every: int, fps: float):
    delay = 1 / fps
    for step in train(config):
        if step.episode % render_every != 0:
            continue
        yield _encode(step, epsilon=step.agent.epsilon)
        await asyncio.sleep(delay)


async def _stream_play(config: PlayConfig, fps: float):
    delay = 1 / fps
    for step in play(config):
        yield _encode(step, epsilon=None)
        await asyncio.sleep(delay)


@app.get("/train")
def stream_train(
    n_episodes: int = Query(50, ge=1, le=200),
    grid_size: int = Query(20, ge=8, le=40),
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay_episodes: int = 5_000,
    render_every: int = Query(1, ge=1),
    fps: float = Query(30.0, gt=0),
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
        _stream_train(config, render_every, fps), media_type="text/event-stream"
    )


@app.get("/play")
def stream_play(
    n_episodes: int = Query(10, ge=1, le=100),
    grid_size: int = Query(20, ge=8, le=40),
    fps: float = Query(10.0, gt=0),
) -> StreamingResponse:
    config = PlayConfig(
        n_episodes=n_episodes, grid_size=grid_size, q_table_path=EXAMPLE_Q_TABLE_PATH
    )
    return StreamingResponse(_stream_play(config, fps), media_type="text/event-stream")
