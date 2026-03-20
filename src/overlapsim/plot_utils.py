from __future__ import annotations

import matplotlib.pyplot as plt


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