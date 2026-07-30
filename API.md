# HTTP API

`api.py` is a separate, optional entry point that exposes `train()`/`play()`
over HTTP as [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
(SSE) — one JSON frame per environment step — so a browser can render a live
training run or a greedy playthrough as an actual snake game. It's the same
`train()`/`play()` generators the CLI uses; nothing about the RL logic is
duplicated.

Neither endpoint writes to disk. `/train` doesn't save a Q-table (the CLI's
`main.py train` still does — see the main [README](README.md)); `/play`
always loads the table committed at `example_q_table.json`, since the real,
gitignored `q_table.json` won't exist in a fresh deployment.

## Running it

```bash
uv sync --extra api
uv run --extra api uvicorn api:app --reload   # http://localhost:8000
```

Deployed to [Railway](https://railway.app) via the included `Procfile`
(`web: uv run --extra api uvicorn api:app --host 0.0.0.0 --port $PORT`) —
Railway supplies `$PORT`.

CORS is wide open (`allow_origins=["*"]`) — there's no auth and nothing
sensitive being served, so any origin can call these endpoints directly from
the browser.

## `GET /train`

Streams a live training run — the agent starts from a fresh, untrained
Q-table and learns as it goes, so early frames look random and it visibly
improves over the course of the stream.

| Param                    | Type  | Default | Range       | Notes                                                |
| ------------------------ | ----- | ------- | ----------- | ---------------------------------------------------- |
| `n_episodes`             | int   | `50`    | `1`–`200`   |                                                      |
| `grid_size`              | int   | `20`    | `8`–`40`    |                                                      |
| `alpha`                  | float | `0.1`   | `0.0`–`1.0` | learning rate                                        |
| `gamma`                  | float | `0.9`   | `0.0`–`1.0` | discount factor                                      |
| `epsilon_start`          | float | `1.0`   | `0.0`–`1.0` | initial exploration rate                             |
| `epsilon_end`            | float | `0.01`  | `0.0`–`1.0` | exploration rate floor                               |
| `epsilon_decay_episodes` | int   | `5000`  | `≥1`        | episodes over which epsilon decays from start to end |
| `render_every`           | int   | `1`     | `≥1`        | only stream every Nth episode (skip the rest)        |
| `fps`                    | float | `30.0`  | `1`–`120`   | frames streamed per second                           |
| `use_shield`             | bool  | `true`  |             | flood-fill safety shield restricting action selection |

## `GET /play`

Streams a greedy playthrough (`epsilon=0`, no learning) using the committed
`example_q_table.json`.

| Param        | Type  | Default | Range     | Notes                      |
| ------------ | ----- | ------- | --------- | -------------------------- |
| `n_episodes` | int   | `10`    | `1`–`100` |                            |
| `grid_size`  | int   | `20`    | `8`–`40`  |                            |
| `fps`        | float | `10.0`  | `1`–`120` | frames streamed per second |
| `use_shield` | bool  | `true`  |           | flood-fill safety shield restricting action selection |

Returns `503` if the Q-table file is missing (a misconfigured deployment) —
checked before the stream starts, so a client never sees a `200` with a
silently empty body.

## Response format

Both endpoints return `Content-Type: text/event-stream`. Every frame is:

```text
data: {"episode": 3, "board": {"grid_size": 20, "snake_body": [[10, 10], [10, 11]], "food": [4, 7]}, "score": 4, "reward": 0, "done": false, "epsilon": 0.87}

```

| Field              | Type            | Notes                                                                   |
| ------------------ | --------------- | ----------------------------------------------------------------------- |
| `episode`          | int             | 0-indexed                                                               |
| `board.grid_size`  | int             |                                                                         |
| `board.snake_body` | `[[x, y], ...]` | tail first, head last                                                   |
| `board.food`       | `[x, y]`        |                                                                         |
| `score`            | int             | snake's current length                                                  |
| `reward`           | float           | this step's reward (`10` on eating food, `-10` on death, `0` otherwise) |
| `done`             | bool            | episode ended this step (real death or starvation timeout)              |
| `epsilon`          | float or `null` | current exploration rate on `/train`; always `null` on `/play`          |

A stream ends when `n_episodes` completes or `15000` frames have been sent,
whichever comes first — that frame cap exists so a single request (e.g. a
long `/play` run on a well-trained agent) can't hold a connection open or
grow a response unboundedly.

## Example: curl

```bash
curl -N "http://localhost:8000/play?n_episodes=3&grid_size=10&fps=5"
```

## Example: browser (`EventSource`)

```js
const source = new EventSource("http://localhost:8000/play?grid_size=10");

source.onmessage = (event) => {
  const frame = JSON.parse(event.data);
  drawBoard(frame.board); // your rendering code
  if (frame.done) {
    console.log("episode over, score:", frame.score);
  }
};
```

`EventSource` only supports `GET` requests with no custom headers or body,
which is why every parameter above is a query string param rather than a
JSON request body.

## Errors

Invalid or out-of-range query params return FastAPI's standard `422` with a
JSON body describing which field failed validation — e.g.
`GET /train?grid_size=4` (below the `8` minimum) or
`GET /play?fps=0` (below the `1` minimum).
