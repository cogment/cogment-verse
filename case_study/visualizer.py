import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Visualizer:
    """Visualization of forcast and anomaly detection results

    Attributes:
        figsize: size of figure
        line_width: Line width
        ms: marker size
        fontsize: Font size
        ndiv_x: Number of division in the x-direction
        ndiv_y: Number of divion in the y-direcition
    """

    def __init__(
        self,
        figsize: tuple = (8, 4),
        line_width: int = 2,
        marker_size: int = 10,
        fontsize: int = 18,
        ndiv_x: int = 5,
        ndiv_y: int = 5,
    ) -> None:
        self.figsize = figsize
        self.line_width = line_width
        self.marker_size = marker_size
        self.fontsize = fontsize
        self.ndiv_x = ndiv_x
        self.ndiv_y = ndiv_y

    def plot_data(
        self,
        axes: plt.Axes,
        x_horz: np.ndarray,
        y_vert: np.ndarray,
        x_label: str,
        y_label: str,
        color: str = "black",
        lgd: Union[str, None] = None,
        file_name: Union[str, None] = None,
        saved_dir: Union[str, None] = None,
    ) -> plt.Axes:
        """Plot 2D data"""
        # Figure
        if axes is None:
            plt.figure(figsize=self.figsize)
            axes = plt.axes()

        # Plot
        axes.plot(x_horz, y_vert, c=color, marker="", linestyle="-", lw=self.line_width, clip_on=False, label=lgd)
        axes.set_xlabel(x_label, fontsize=self.fontsize)
        axes.set_ylabel(y_label, fontsize=self.fontsize)

        # Lengend
        if lgd is not None:
            axes.legend(loc="best", edgecolor=None, fontsize=0.9 * self.fontsize, ncol=1, framealpha=0.3)

        # Save pdf file
        if saved_dir is not None:
            os.makedirs(saved_dir, exist_ok=True)
            saving_path = f"{saved_dir}/{file_name}.png"
            plt.savefig(saving_path, bbox_inches="tight")
            plt.close()

        return axes

    def plot_reward(
        self,
        num_steps: List[np.ndarray],
        rewards: List[np.ndarray],
        labels: List[str],
        file_name: Union[str, None] = None,
        saved_dir: Union[str, None] = None,
    ) -> None:
        """Plot reward"""
        max_step = max(np.amax(array) for array in num_steps)
        min_step = min(np.amin(array) for array in num_steps)
        max_reward = max(np.amax(array) for array in rewards)
        max_reward = max_reward + 0.1 * max_reward
        min_reward = min(np.amin(array) for array in rewards)

        x_label = "Number of steps (M)"
        y_label = "Average of trial steps"

        plt.figure(figsize=self.figsize)
        axe = plt.axes()

        colors = plt.cm.viridis(np.linspace(0, 1, len(num_steps)))
        for num_step, reward, lgd, color in zip(num_steps, rewards, labels, colors):
            axe = self.plot_data(
                axes=axe, x_label=x_label, y_label=y_label, x_horz=num_step, y_vert=reward, lgd=lgd, color=color
            )

        x_ticks = np.linspace(min_step, max_step, self.ndiv_x, endpoint=True)
        y_ticks = np.linspace(min_reward, max_reward, self.ndiv_x, endpoint=True)
        axe.set_yticks(y_ticks)
        axe.set_xticks(x_ticks)
        axe.set_ylim(min_reward, max_reward)
        axe.set_xlim(min_step, max_step)
        axe.set_xlabel(x_label, fontsize=self.fontsize)
        axe.set_ylabel(y_label, fontsize=self.fontsize)
        axe.tick_params(axis="both", which="both", direction="inout", labelsize=self.fontsize)

        # Save pdf file
        if saved_dir is not None:
            os.makedirs(saved_dir, exist_ok=True)
            saving_path = f"{saved_dir}/{file_name}.png"
            plt.savefig(saving_path, bbox_inches="tight")
            plt.close()


def load_csv(file_path: str) -> pd.DataFrame:
    """Load csv"""
    data_frame = pd.read_csv(file_path)
    return data_frame


def main() -> None:
    """Plot API"""
    project_dir = "./case_study"
    saved_dir = f"{project_dir}/figure"
    file_name = "reward_comp"
    target_reward = 4000

    num_steps = []
    rewards = []
    labels = []
    target_len = 0
    for i in range(3):
        file_path = f"{project_dir}/metrics_{i}.csv"
        data_frame = load_csv(file_path=file_path)
        num_step = data_frame["step"].values * 512
        reward = data_frame["value"].values
        if len(num_step) > target_len:
            target_num_step = num_step / 1e6
            target_len = len(num_step)

        num_steps.append(num_step / 1e6)
        rewards.append(reward)
        labels.append(f"run_{i}")

    # Target
    num_steps.append(target_num_step)
    rewards.append(target_num_step * 0 + target_reward)
    labels.append("Target")

    num_steps.append(target_num_step)
    rewards.append(target_num_step * 0 + 1400)
    labels.append("Previous runs")

    # Visualizatiom
    viz = Visualizer()
    viz.plot_reward(num_steps=num_steps, rewards=rewards, labels=labels, file_name=file_name, saved_dir=saved_dir)


if __name__ == "__main__":
    main()
