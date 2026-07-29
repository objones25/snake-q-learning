from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_progress(
    checkpoints: list[tuple[int, float]], save_path: Path | None = None
) -> None:
    """Plot rolling-average score over the course of training.

    `checkpoints` is a list of (episode, avg_score) pairs, e.g. the same
    500-episode rolling average main.py already prints during training.
    """
    episodes, avg_scores = zip(*checkpoints)
    plt.figure()
    plt.plot(episodes, avg_scores)
    plt.xlabel("Episode")
    plt.ylabel("Avg score (rolling, last 500 episodes)")
    plt.title("Training progress")
    _finish(save_path)


def plot_score_distribution(scores: list[int], save_path: Path | None = None) -> None:
    """Plot a histogram of per-episode scores, e.g. from a play run."""
    plt.figure()
    # One bin per integer score. A fixed bin count (e.g. 30) doesn't divide
    # evenly into an integer score range, so some bins catch more distinct
    # score values than their neighbors — that shows up as spiky bars even
    # when the underlying distribution is smooth.
    bins = range(min(scores), max(scores) + 2)
    plt.hist(scores, bins=bins)
    plt.xlabel("Score")
    plt.ylabel("Episodes")
    plt.title("Score distribution")
    _finish(save_path)


def _finish(save_path: Path | None) -> None:
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
