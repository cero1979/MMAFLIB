"""Interactive field mowing demo using ipywidgets.

This module exposes :func:`podar_campo_demo`, an interactive demonstration
that compares the areas of an outer square of side ``b`` with an inner square
produced by mowing a strip of width ``x`` on each side.  Sliders let the user
adjust ``b`` and ``x`` in Jupyter/Colab environments without needing a special
Matplotlib backend.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import clear_output, display


def podar_campo_demo(b0: float = 10.0, x0: float = 2.0):
    """Display an interactive plot showing the mowed area of a square field.

    Parameters
    ----------
    b0 : float, default 10.0
        Initial side length of the outer square.
    x0 : float, default 2.0
        Initial width of the mowed strip on each side.
    """

    out = widgets.Output()

    b_slider = widgets.FloatSlider(value=b0, min=0.1, max=20.0, step=0.1, description="b")
    x_slider = widgets.FloatSlider(value=x0, min=0.0, max=b0 / 2, step=0.1, description="x")

    def draw(b: float, x: float) -> None:
        """Redraw the field for the current slider values."""
        if x > b / 2:
            x = b / 2
            x_slider.value = x

        inner_side = b - 2 * x
        b_slider.min = inner_side
        x_slider.max = b / 2

        fig, ax_field = plt.subplots(figsize=(6, 6))
        ax_field.set_aspect("equal")
        ax_field.axis("off")

        outer_sq = plt.Rectangle((0, 0), b, b, facecolor="#d5f2c2", edgecolor="black")
        inner_sq = plt.Rectangle((x, x), inner_side, inner_side,
                                 facecolor="white", edgecolor="black")
        ax_field.add_patch(outer_sq)
        ax_field.add_patch(inner_sq)
        ax_field.set_xlim(0, b)
        ax_field.set_ylim(0, b)

        A1 = b * b
        A2 = inner_side * inner_side
        diff = A1 - A2
        formula = 4 * x * (b - x)
        text = (
            f"b = {b:.2f}\n"
            f"x = {x:.2f}\n"
            f"A₁ = b² = {A1:.2f}\n"
            f"A₂ = (b - 2x)² = {A2:.2f}\n"
            f"A₁ - A₂ = {diff:.2f}\n"
            f"4x(b - x) = {formula:.2f}"
        )
        ax_field.text(1.05, 1, text, transform=ax_field.transAxes,
                      va="top", ha="left")

        with out:
            clear_output(wait=True)
            display(fig)
        plt.close(fig)

    def on_change(change):
        draw(b_slider.value, x_slider.value)

    b_slider.observe(on_change, names="value")
    x_slider.observe(on_change, names="value")
    draw(b0, x0)

    controls = widgets.HBox([b_slider, x_slider])
    display(widgets.VBox([out, controls]))

    return out, (b_slider, x_slider)


if __name__ == "__main__":
    podar_campo_demo()
