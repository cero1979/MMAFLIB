import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Initial parameters
b0 = 10.0  # Outer square side length
x0 = 2.0   # Width of mowed stripe

# Create figure
fig = plt.figure(figsize=(8, 6))

# Axes for drawing the field
ax_field = fig.add_axes([0.1, 0.25, 0.5, 0.7])
ax_field.set_aspect('equal')
ax_field.axis('off')

# Draw outer and inner squares
outer_square = plt.Rectangle((0, 0), b0, b0,
                              facecolor='#d5f2c2', edgecolor='black')
inner_side = b0 - 2 * x0
inner_square = plt.Rectangle((x0, x0), inner_side, inner_side,
                              facecolor='white', edgecolor='black')
ax_field.add_patch(outer_square)
ax_field.add_patch(inner_square)
ax_field.set_xlim(0, b0)
ax_field.set_ylim(0, b0)

# Text-only axes to display parameters
ax_text = fig.add_axes([0.65, 0.25, 0.3, 0.7])
ax_text.axis('off')
text_box = ax_text.text(0, 1, '', va='top', ha='left')

# Slider axes positioned below both axes
ax_slider_b = fig.add_axes([0.1, 0.15, 0.8, 0.03])
slider_b = Slider(ax_slider_b, 'b', inner_side, 20, valinit=b0)

ax_slider_x = fig.add_axes([0.1, 0.1, 0.8, 0.03])
slider_x = Slider(ax_slider_x, 'x', 0, b0 / 2, valinit=x0)


def update_text(b, x, inner_side):
    A1 = b * b
    A2 = inner_side * inner_side
    podada_diff = A1 - A2
    podada_formula = 4 * x * (b - x)
    text_box.set_text(
        f"b = {b:.2f}\n"
        f"x = {x:.2f}\n"
        f"A₁ = b² = {A1:.2f}\n"
        f"A₂ = (b - 2x)² = {A2:.2f}\n"
        f"A₁ - A₂ = {podada_diff:.2f}\n"
        f"4x(b - x) = {podada_formula:.2f}"
    )
    fig.canvas.draw_idle()


def update_b(val):
    b = slider_b.val
    inner_side = inner_square.get_width()
    if b < inner_side:
        slider_b.eventson = False
        slider_b.set_val(inner_side)
        slider_b.eventson = True
        b = inner_side
    outer_square.set_width(b)
    outer_square.set_height(b)
    ax_field.set_xlim(0, b)
    ax_field.set_ylim(0, b)
    x = (b - inner_side) / 2
    inner_square.set_xy((x, x))
    slider_x.eventson = False
    slider_x.valmax = b / 2
    slider_x.set_val(x)
    slider_x.eventson = True
    update_text(b, x, inner_side)


def update_x(val):
    x = slider_x.val
    b = slider_b.val
    if x > b / 2:
        x = b / 2
        slider_x.eventson = False
        slider_x.set_val(x)
        slider_x.eventson = True
    inner_side = b - 2 * x
    inner_square.set_width(inner_side)
    inner_square.set_height(inner_side)
    inner_square.set_xy((x, x))
    slider_b.eventson = False
    slider_b.valmin = inner_side
    slider_b.ax.set_xlim(slider_b.valmin, slider_b.valmax)
    if slider_b.val < inner_side:
        slider_b.set_val(inner_side)
    slider_b.eventson = True
    update_text(b, x, inner_side)


slider_b.on_changed(update_b)
slider_x.on_changed(update_x)

# Initialize display
update_x(x0)

plt.show()