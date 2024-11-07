import math

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_colourtable(colours, name):
    """Plot colour table.

    Inspired by https://matplotlib.org/stable/gallery/color/named_colors.html

    Args:
        colors (list[str]): List of colour hex codes

    """

    matplotlib.rcParams["font.family"] = "quicksand"
    matplotlib.rcParams["mathtext.fontset"] = "quicksand"

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    ncols = 2

    n = len(colours)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(
        margin / width,
        margin / height,
        (width - margin) / width,
        (height - margin) / height,
    )
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, colour in enumerate(colours):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(
            text_pos_x,
            y,
            colour,
            fontsize=14,
            horizontalalignment="left",
            verticalalignment="center",
        )

        ax.add_patch(
            Rectangle(
                xy=(swatch_start_x, y - 9),
                width=swatch_width,
                height=18,
                facecolor=colour,
                # edgecolor="0.7",
            )
        )

    # return fig
    for ext in ["pdf", "svg", "png"]:
        fname = f"{name}.colourtable.{ext}"
        print(f"Saving figure to file '{fname}'")
        fig.savefig(fname, dpi=300)
    
    plt.close(fig)


def main():
    pennylane_colours = [
        "#00b1ff",  # blue
        "#ff00d9",  # magenta
        "#ffbe0c",  # yellow
        "#53585f",  # grey
        "#b3e2f7",  # pale blue
        "#ffcff8",  # pale magenta
        "#ffe9ad",  # pale yellow
        "#9e9e9e",  # pale grey
    ]

    plot_colourtable(pennylane_colours, "pennylane")


if __name__ == "__main__":
    main()
