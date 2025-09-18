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

    import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import random

def generar_escenario():
    # Generar punto de intersección entero aleatorio en cada ejecución
    C_real = np.array([random.randint(-2, 4), random.randint(0, 4)])

    # Generar posiciones de drones enteras, distintas de C_real y separadas de C_real
    def random_drone_point(exclude, xlim, ylim):
        while True:
            p = np.array([random.randint(*xlim), random.randint(*ylim)])
            if not np.array_equal(p, exclude):
                return p

    D1 = random_drone_point(C_real, (-5, C_real[0]-1), (-2, C_real[1]-1))
    D2 = random_drone_point(C_real, (C_real[0]+1, 7), (C_real[1]+1, 5))

    v1 = C_real - D1
    v2 = C_real - D2
    S1 = C_real + 0.8 * v1
    S2 = C_real + 0.8 * v2

    return C_real, D1, D2, v1, v2, S1, S2

def evitar_colision_factory():
    C_real, D1, D2, v1, v2, S1, S2 = generar_escenario()

    def evitar_colision(x_c, y_c):
        C = np.array([x_c, y_c])
        distancia = np.linalg.norm(C - C_real)
        tolerancia = 0.05

        t_vals = np.linspace(-0.2, 1.4, 200)
        l1_x = D1[0] + t_vals * v1[0]
        l1_y = D1[1] + t_vals * v1[1]
        l2_x = D2[0] + t_vals * v2[0]
        l2_y = D2[1] + t_vals * v2[1]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(l1_x, l1_y, color="dodgerblue", linewidth=2, label="Trayectoria dron 1")
        ax.plot(l2_x, l2_y, color="darkorange", linewidth=2, label="Trayectoria dron 2")
        ax.scatter(*D1, color="dodgerblue", s=80, label="Dron 1")
        ax.scatter(*D2, color="darkorange", s=80, label="Dron 2")
        ax.scatter(*C, color="purple", marker="*", s=200, label=fr"Punto propuesto C({x_c}, {y_c})")
        # Mostrar el punto de intersección real sin taparlo con la leyenda
        ax.scatter(*C_real, color="black", marker="x", s=100, zorder=10, label="Punto de cruce real")

        if distancia <= tolerancia:
            ax.scatter(*S1, color="navy", marker="s", s=120, label=fr"Punto extra dron 1 S₁({S1[0]:.2f}, {S1[1]:.2f})")
            ax.scatter(*S2, color="sienna", marker="s", s=120, label=fr"Punto extra dron 2 S₂({S2[0]:.2f}, {S2[1]:.2f})")

        ax.set_xlim(-6, 7)
        ax.set_ylim(-3, 6)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        # Colocar la leyenda fuera del área de la gráfica para no tapar el cruce
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1))
        ax.set_title("Localiza el punto de cruce de los drones (coordenadas enteras)")
        plt.show()

        if distancia <= tolerancia:
            print("✅ Coordenadas correctas del cruce (enteras).")
            print(f"C = ({C_real[0]}, {C_real[1]})")
            print(f"S₁ = ({S1[0]:.2f}, {S1[1]:.2f})   (sobre la recta del dron 1)")
            print(f"S₂ = ({S2[0]:.2f}, {S2[1]:.2f})   (sobre la recta del dron 2)")
        else:
            print("Ajusta C: los puntos extra S₁ y S₂ aparecerán cuando aciertes el cruce real.")

    return evitar_colision

def lanzar_interactivo():
    interact(
        evitar_colision_factory(),
        x_c=IntSlider(value=0, min=-2, max=5, step=1, description="x de C"),
        y_c=IntSlider(value=0, min=0, max=5, step=1, description="y de C")
    )