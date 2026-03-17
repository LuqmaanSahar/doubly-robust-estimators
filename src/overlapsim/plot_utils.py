from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_gray_style(
    ax,
    facecolor: str = "#E6E6E6",
    grid_color: str = "w",
):
    """
    Applies my custom style to axes.

    Parameters
    ----------
    ax : plt.Axes or None, default None
        Axes to style.
        If None, uses the current axes from matplotlib.
    facecolor : str, default "#E6E6E6"
        Background color to apply to the axes.
    grid_color : str, default "w"
        Color to use for the gridlines.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        ax = plt.gca()

    # background
    ax.set_facecolor(facecolor)
    ax.set_axisbelow(True)

    # hide spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # add gridlines
    ax.grid(True, color=grid_color, linestyle="solid")

    # ticks and labels
    ax.tick_params(colors="gray", direction="out")
    for tick in ax.get_xticklabels():
        tick.set_color("gray")
    for tick in ax.get_yticklabels():
        tick.set_color("gray")

    return ax



def plot_overlap(
    summary_df: pd.DataFrame,
    metric: str = "mean_cate_rmse",
) -> None:
    """
    Plot a summary metric against overlap scale.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary_df["overlap_scale"], summary_df[metric], marker="o")
    ax.set_xlabel("Overlap scale")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by overlap scale")
    plt.tight_layout()
    plt.show()


def plot_metric_grid(
    summary_df: pd.DataFrame,
    metric: str = "mean_cate_rmse",
) -> None:
    """
    Plot a summary metric against treatment prevalence,
    with one line per overlap level.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for overlap_scale, group in summary_df.groupby("overlap_scale"):
        group = group.sort_values("target_treatment_rate")
        ax.plot(
            group["target_treatment_rate"],
            group[metric],
            marker="o",
            label=f"overlap={overlap_scale}",
        )

    ax.set_xlabel("Target treatment rate")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across overlap and treatment rate")
    ax.legend()
    plt.tight_layout()
    plt.show()