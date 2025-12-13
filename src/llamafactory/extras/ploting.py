# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
from typing import Any

from transformers.trainer import TRAINER_STATE_NAME

from . import logging
from .packages import is_matplotlib_available


if is_matplotlib_available():
    import matplotlib.figure
    import matplotlib.pyplot as plt


logger = logging.get_logger(__name__)


def smooth(scalars: list[float]) -> list[float]:
    r"""EMA implementation according to TensorBoard."""
    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def gen_loss_plot(trainer_log: list[dict[str, Any]]) -> "matplotlib.figure.Figure":
    r"""Plot loss curves in LlamaBoard."""
    plt.close("all")
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    steps, losses = [], []
    for log in trainer_log:
        if log.get("loss", None):
            steps.append(log["current_steps"])
            losses.append(log["loss"])

    ax.plot(steps, losses, color="#1f77b4", alpha=0.4, label="original")
    ax.plot(steps, smooth(losses), color="#1f77b4", label="smoothed")
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    return fig


def plot_loss(save_dictionary: str, keys: list[str] = ["loss"]) -> None:
    r"""Plot loss curves and saves the image in a single figure when multiple keys are provided."""
    plt.switch_backend("agg")
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), encoding="utf-8") as f:
        data = json.load(f)

    series: dict[str, tuple[list[int], list[float]]] = {}
    for key in keys:
        steps, metrics = [], []
        for log in data.get("log_history", []):
            if key in log:
                steps.append(log["step"])
                metrics.append(log[key])

        if len(metrics) == 0:
            logger.warning_rank0(f"No metric {key} to plot.")
            continue
        series[key] = (steps, metrics)

    if len(series) == 0:
        return

    plt.figure()
    for key, (steps, metrics) in series.items():
        plt.plot(steps, metrics, alpha=0.35, label=f"{key} (raw)")
        plt.plot(steps, smooth(metrics), label=f"{key} (smoothed)")

    plt.title(f"training metrics of {save_dictionary}")
    plt.xlabel("step")
    plt.ylabel("value")
    plt.legend()
    figure_path = os.path.join(save_dictionary, "training_losses.png")
    plt.savefig(figure_path, format="png", dpi=100)
    print("Figure saved at:", figure_path)
