


"""Interactive field mowing demo using ipywidgets.

This module exposes :func:`podar_campo_demo`, an interactive demonstration
that compares the areas of an outer square of side ``b`` with an inner square
produced by mowing a strip of width ``x`` on each side. Sliders let the user
adjust ``b`` and ``x`` in Jupyter/Colab environments without needing a special
Matplotlib backend.
"""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import clear_output, display
from ipywidgets import IntSlider, interact


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

import numpy as np
def generar_escenario():
    """Generar un escenario aleatorio de cruce de drones."""
    C_real = np.array([random.randint(-2, 4), random.randint(0, 4)])

    def random_drone_point(exclude, xlim, ylim):
        while True:
            p = np.array([random.randint(*xlim), random.randint(*ylim)])
            if not np.array_equal(p, exclude):
                return p

    D1 = random_drone_point(C_real, (-5, C_real[0] - 1), (-2, C_real[1] - 1))
    D2 = random_drone_point(C_real, (C_real[0] + 1, 7), (C_real[1] + 1, 5))

    v1 = C_real - D1
    v2 = C_real - D2
    S1 = C_real + 0.8 * v1
    S2 = C_real + 0.8 * v2

    return C_real, D1, D2, v1, v2, S1, S2


def evitar_colision_factory():
    """Crear una función interactiva para evitar la colisión de drones."""
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
        ax.scatter(*C_real, color="black", marker="x", s=100, zorder=10, label="Punto de cruce real")

        if distancia <= tolerancia:
            ax.scatter(*S1, color="navy", marker="s", s=120,
                       label=fr"Punto extra dron 1 S₁({S1[0]:.2f}, {S1[1]:.2f})")
            ax.scatter(*S2, color="sienna", marker="s", s=120,
                       label=fr"Punto extra dron 2 S₂({S2[0]:.2f}, {S2[1]:.2f})")

        ax.set_xlim(-6, 7)
        ax.set_ylim(-3, 6)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
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
    """Mostrar los controles interactivos del ejercicio de drones."""
    return interact(
        evitar_colision_factory(),
        x_c=IntSlider(value=0, min=-2, max=5, step=1, description="x de C"),
        y_c=IntSlider(value=0, min=0, max=5, step=1, description="y de C"),
    )


def main():
    """Entry point para ejecutar el módulo como script."""
    lanzar_interactivo()

# ===================================================================
# 1. IMPORTACIONES (Combinación de AMBOS scripts)
# ===================================================================
from __future__ import annotations
import random
from fractions import Fraction
from typing import Iterable, List, Sequence, Tuple

import ipywidgets as widgets
from IPython.display import Markdown, display, clear_output, HTML
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, FloatSlider, Button, VBox, HBox, Label, FloatText, Output, IntSlider

# Configuración de estilo de gráficos
plt.style.use('seaborn-v0_8-darkgrid')


# ===================================================================
# 2. DEFINICIONES: DIVISIÓN SINTÉTICA
# ===================================================================

def _formatear(valor: Fraction | None) -> str:
    """Formatea un número (posiblemente Fracción) como cadena."""
    if valor is None:
        return ""
    if isinstance(valor, Fraction):
        if valor.denominator == 1:
            return str(valor.numerator)
        return f"{valor.numerator}/{valor.denominator}"
    return str(valor)


def _parsear_coeficientes(texto: str, grado: int) -> List[Fraction]:
    """Convierte una cadena de texto en una lista de Fracciones."""
    partes = texto.replace(",", " ").split()
    if len(partes) != grado + 1:
        raise ValueError(
            f"Se esperaban {grado + 1} coeficientes y se recibieron {len(partes)}."
        )
    try:
        return [Fraction(parte) for parte in partes]
    except ValueError as exc:
        raise ValueError(
            "Alguno de los coeficientes no es válido. Usa enteros, fracciones (a/b) o decimales."
        ) from exc


def division_sintetica(
    coeficientes: Sequence[Fraction], r: Fraction
) -> Tuple[List[Fraction], List[Fraction | None], List[Fraction]]:
    """Ejecuta el algoritmo de división sintética."""
    productos: List[Fraction | None] = [None] * len(coeficientes)
    acumulados: List[Fraction] = [Fraction(0)] * len(coeficientes)

    for indice, coef in enumerate(coeficientes):
        suma = coef
        if indice > 0 and productos[indice] is not None:
            suma += productos[indice]
        acumulados[indice] = suma
        if indice < len(coeficientes) - 1:
            productos[indice + 1] = suma * r

    return list(coeficientes), productos, acumulados


def _tabla_html(
    encabezado: Iterable[str],
    filas: Iterable[Iterable[str]],
) -> str:
    """Genera una tabla HTML simple."""
    def celda(elemento: str, tag: str = "td") -> str:
        return f"<{tag} style='padding:6px 10px;text-align:center;border:1px solid #bbb;'>{elemento}</{tag}>"

    filas_html = ["<tr>" + "".join(celda(valor) for valor in fila) + "</tr>" for fila in filas]
    encabezados_html = "<tr>" + "".join(celda(valor, "th") for valor in encabezado) + "</tr>"
    return (
        "<table style='border-collapse:collapse;font-family:Segoe UI, sans-serif;margin-top:10px;'>"
        + encabezados_html
        + "".join(filas_html)
        + "</table>"
    )


def _construir_tabla(
    coeficientes: Sequence[Fraction],
    productos: Sequence[Fraction | None],
    acumulados: Sequence[Fraction],
) -> str:
    """Construye la tabla HTML específica para la división sintética."""
    grados = [f"x^{i}" for i in range(len(coeficientes) - 1, -1, -1)]
    encabezado = ["Columna"] + grados
    fila_coef = ["Coeficientes"] + [_formatear(c) for c in coeficientes]
    fila_prod = ["Multiplicaciones"] + ["" if p is None else _formatear(p) for p in productos]
    fila_acum = ["Suma parcial"] + [_formatear(a) for a in acumulados]
    return _tabla_html(encabezado, [fila_coef, fila_prod, fila_acum])


def _describir_resultado(acumulados: Sequence[Fraction]) -> str:
    """Genera el texto de resumen (cociente y resto)."""
    if not acumulados:
        return "No hay datos para mostrar."
    resto = acumulados[-1]
    cociente = acumulados[:-1]
    texto_cociente = ", ".join(_formatear(c) for c in cociente) or "0"
    return (
        f"**Coeficientes del cociente (de x^(n-1) a x^0):** {texto_cociente}\n\n"
        f"**Resto:** {_formatear(resto)}"
    )


def crear_interfaz_division_sintetica() -> None:
    """Construye y muestra los controles interactivos para la división sintética."""

    estilo = {"description_width": "initial"}
    grado_input = widgets.BoundedIntText(
        value=3,
        min=0,
        max=20,
        description="Grado del polinomio",
        style=estilo,
    )
    coef_input = widgets.Textarea(
        value="1 -16 -60 432",
        description="Coeficientes (de x^n a x^0)",
        layout=widgets.Layout(width="100%", height="80px"),
        style=estilo,
    )
    r_input = widgets.Text(
        value="4",
        description="Valor de r (x - r)",
        style=estilo,
    )
    boton = widgets.Button(description="Calcular tabla", button_style="success", icon="calculator")
    salida = widgets.Output()

    def _calcular(_=None) -> None:
        """Función interna llamada por el botón."""
        with salida:
            salida.clear_output()
            try:
                grado = grado_input.value
                coeficientes = _parsear_coeficientes(coef_input.value, grado)
                r = Fraction(r_input.value)
            except ValueError as error:
                display(Markdown(f"**Error:** {error}"))
                return

            coef, productos, acumulados = division_sintetica(coeficientes, r)
            tabla_html = _construir_tabla(coef, productos, acumulados)
            resumen = _describir_resultado(acumulados)

            display(Markdown("**Tabla de división sintética:**"))
            display(widgets.HTML(value=tabla_html))
            display(Markdown(resumen))

    boton.on_click(_calcular)

    display(
        widgets.VBox(
            [
                widgets.HTML("<h3>División sintética interactiva</h3>"),
                widgets.HTML(
                    "<p>Introduce el grado, los coeficientes de <em>P(x)</em> y el valor <em>r</em> del divisor <strong>(x - r)</strong>."),
                grado_input,
                coef_input,
                r_input,
                boton,
                salida,
            ]
        )
    )

    # Ejecutar un cálculo inicial al mostrar
    _calcular()


# ===================================================================
# 3. DEFINICIONES: DEMO FLEXIÓN DE VIGA
# ===================================================================

# --- Funciones de cálculo ---

def deflection(x):
    """
    Calcula la deflexión de la viga en la posición x
    d(x) = (1/16000)(60x² - x³)
    """
    return (1/16000) * (60 * x**2 - x**3)


def deflection_derivative(x):
    """
    Calcula la derivada de d(x)
    d'(x) = (1/16000)(120x - 3x²)
    """
    return (1/16000) * (120*x - 3*x**2)

# --- Función de ploteo ---

def plot_beam_deflection(x_position=10.0):
    """
    Grafica la curva de deflexión y muestra el punto seleccionado
    """
    x = np.linspace(0, 20, 500)
    y = deflection(x)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Gráfica principal
    ax1.plot(x, y, 'b-', linewidth=2.5, label='Curva de flexión d(x)')
    ax1.axvline(x_position, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Posición x = {x_position:.2f} pies')
    ax1.plot(x_position, deflection(x_position), 'ro', markersize=12, label=f'd = {deflection(x_position):.4f} pulg')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Posición x (pies)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Flexión d (pulgadas)', fontsize=12, fontweight='bold')
    ax1.set_title('Flexión de la Viga en Voladizo', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 20)
    
    # Diagrama esquemático de la viga
    ax2.set_xlim(-2, 22)
    ax2.set_ylim(-3, 2)
    ax2.axis('off')
    
    # Dibujar la viga
    viga_x = np.linspace(0, 20, 100)
    viga_y = -deflection(viga_x) * 0.3  # Escalar para visualización
    ax2.plot(viga_x, viga_y, 'brown', linewidth=8, solid_capstyle='round')
    
    # Soporte fijo (pared)
    ax2.fill_between([-1, 0], [-2, -2], [2, 2], color='gray', alpha=0.5, hatch='///')
    ax2.plot([0, 0], [-2, 2], 'k-', linewidth=3)
    
    # Carga en el extremo
    ax2.arrow(20, -deflection(20)*0.3, 0, -1.5, head_width=0.5, head_length=0.3, 
              fc='darkred', ec='darkred', linewidth=2)
    ax2.text(20, -deflection(20)*0.3 - 2, '600 lb', ha='center', fontsize=12, 
             fontweight='bold', color='darkred')
    
    # Indicador de posición actual
    ax2.plot(x_position, -deflection(x_position)*0.3, 'ro', markersize=15, zorder=5)
    ax2.arrow(x_position, 1.5, 0, -1.3 + deflection(x_position)*0.3, 
              head_width=0.3, head_length=0.2, fc='red', ec='red', linewidth=1.5)
    ax2.text(x_position, 1.8, f'x = {x_position:.1f} ft', ha='center', fontsize=11, 
             fontweight='bold', color='red')
    
    # Dimensión total
    ax2.plot([0, 20], [-2.5, -2.5], 'k-', linewidth=1)
    ax2.plot([0, 0], [-2.5, -2.7], 'k-', linewidth=1)
    ax2.plot([20, 20], [-2.5, -2.7], 'k-', linewidth=1)
    ax2.text(10, -2.9, '20 pies', ha='center', fontsize=11, fontweight='bold')
    
    ax2.set_title('Diagrama Esquemático', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📍 Posición seleccionada: x = {x_position:.3f} pies")
    print(f"📏 Flexión calculada: d = {deflection(x_position):.6f} pulgadas")


# --- Función principal para lanzar la demo de la viga ---

def lanzar_demo_viga():
    """Crea y muestra la interfaz completa de la demo de flexión de viga."""
    
    # Variables globales para las respuestas
    respuestas = {'p1': None, 'p2': None, 'p3a': None, 'p3b': None}

    # Funciones de verificación (internas a esta demo)
    def verificar_pregunta1(b):
        with output1:
            clear_output()
            x_resp = input1_x.value
            d_resp = input1_d.value
            
            x_correcto = 20.0
            d_correcto = deflection(20.0)
            
            tol_x = 0.5
            tol_d = 0.05
            
            if abs(x_resp - x_correcto) < tol_x and abs(d_resp - d_correcto) < tol_d:
                print("✅ ¡CORRECTO! Excelente trabajo.")
                print(f"    La flexión máxima ocurre en x = {x_correcto} pies")
                print(f"    con un valor de d_max = {d_correcto:.4f} pulgadas")
                respuestas['p1'] = True
            else:
                print("❌ No es correcto. Intenta de nuevo.")
                print("    Usa el explorador interactivo para encontrar el punto más alto de la curva.")
                if abs(x_resp - x_correcto) >= tol_x:
                    print(f"    Tu valor de x está {'muy lejos' if abs(x_resp - x_correcto) > 2 else 'cerca'}.")

    def verificar_pregunta2(b):
        with output2:
            clear_output()
            d_resp = input2.value
            d_correcto = deflection(10.0)
            
            tol = 0.01
            
            if abs(d_resp - d_correcto) < tol:
                print("✅ ¡CORRECTO!")
                print(f"    En la mitad de la viga (x = 10 pies), la flexión es d = {d_correcto:.4f} pulgadas")
                respuestas['p2'] = True
            else:
                print("❌ No es correcto. Intenta de nuevo.")
                print("    Usa el slider para posicionar en x = 10 y observa el valor de d.")

    def verificar_pregunta3(b):
        with output3:
            clear_output()
            x1_resp = input3a.value
            x2_resp = input3b.value
            
            soluciones_correctas = []
            for x_test in np.linspace(0, 20, 10000):
                if abs(deflection(x_test) - 0.5) < 0.001:
                    if not any(abs(x_test - sol) < 0.5 for sol in soluciones_correctas):
                        soluciones_correctas.append(x_test)
            
            tol = 0.3
            respuestas_usuario = [x1_resp, x2_resp]
            
            correctas = 0
            for x_resp in respuestas_usuario:
                if x_resp > 0:  # Solo considerar valores ingresados
                    for x_correct in soluciones_correctas:
                        if abs(x_resp - x_correct) < tol:
                            correctas += 1
                            break
            
            if correctas == len(soluciones_correctas) and correctas > 0:
                print("✅ ¡CORRECTO! Has encontrado todas las soluciones.")
                print(f"    Posiciones donde d = 0.5 pulgadas:")
                for i, sol in enumerate(soluciones_correctas, 1):
                    print(f"        x{i} ≈ {sol:.2f} pies")
                respuestas['p3a'] = True
                respuestas['p3b'] = True
            elif correctas > 0:
                print("⚠️ Parcialmente correcto.")
                print(f"    Has encontrado {correctas} de {len(soluciones_correctas)} solución(es).")
                print("    Sigue explorando con el slider.")
            else:
                print("❌ No es correcto. Intenta de nuevo.")
                print("    Mueve el slider lentamente y busca donde d ≈ 0.5")

    # --- Construcción de la interfaz ---
    
    print("="*60)
    print("PROBLEMA: FLEXIÓN DE UNA VIGA EN VOLADIZO")
    print("="*60)
    print("\nUna viga en voladizo tiene 20 pies de longitud y se carga")
    print("con 600 lb en su extremo derecho. La flexión está dada por:")
    print("\n    d(x) = (1/16000)(60x² - x³)")
    print("\ndonde:")
    print("  - d: flexión en pulgadas")
    print("  - x: distancia desde el soporte en pies (0 ≤ x ≤ 20)")
    print("="*60)

    print("\n\n" + "="*60)
    print("EXPLORADOR INTERACTIVO")
    print("="*60)
    print("Mueve el slider para explorar la deflexión en diferentes posiciones\n")

    slider_widget = interactive(plot_beam_deflection, 
                                x_position=FloatSlider(min=0, max=20, step=0.1, value=10, 
                                                       description='Posición x:', 
                                                       style={'description_width': '100px'},
                                                       continuous_update=True))
    display(slider_widget)

    print("\n\n" + "="*60)
    print("PREGUNTAS A RESOLVER")
    print("="*60)

    # Pregunta 1
    print("\n📌 PREGUNTA 1: Flexión máxima")
    print("-" * 60)
    print("¿En qué posición x (en pies) se produce la flexión MÁXIMA?")
    print("¿Cuál es el valor de esa flexión máxima (en pulgadas)?")
    print("\nPista: Observa el punto más alto de la curva en la gráfica.")
    
    output1 = Output()
    input1_x = FloatText(description='x =', style={'description_width': '50px'}, step=0.01)
    input1_d = FloatText(description='d_max =', style={'description_width': '50px'}, step=0.001)
    button1 = Button(description='Verificar Respuesta 1', button_style='info')
    button1.on_click(verificar_pregunta1)

    print("\nIngresa tus respuestas:")
    display(HBox([Label('Posición:'), input1_x, Label('pies')]))
    display(HBox([Label('Flexión máxima:'), input1_d, Label('pulgadas')]))
    display(button1)
    display(output1)

    # Pregunta 2
    print("\n\n📌 PREGUNTA 2: Flexión en la mitad de la viga")
    print("-" * 60)
    print("¿Cuál es la flexión (en pulgadas) cuando x = 10 pies?")
    print("(Es decir, en la mitad de la viga)")

    output2 = Output()
    input2 = FloatText(description='d(10) =', style={'description_width': '60px'}, step=0.001)
    button2 = Button(description='Verificar Respuesta 2', button_style='info')
    button2.on_click(verificar_pregunta2)

    print("\nIngresa tu respuesta:")
    display(HBox([input2, Label('pulgadas')]))
    display(button2)
    display(output2)

    # Pregunta 3
    print("\n\n📌 PREGUNTA 3: Posiciones con flexión específica")
    print("-" * 60)
    print("¿En qué posición(es) x la flexión es igual a 0.5 pulgadas?")
    print("Nota: Puede haber MÁS DE UNA respuesta. Si encuentras dos valores, ingrésalos ambos.")

    output3 = Output()
    input3a = FloatText(description='x₁ =', style={'description_width': '50px'}, step=0.01)
    input3b = FloatText(description='x₂ =', style={'description_width': '50px'}, step=0.01)
    button3 = Button(description='Verificar Respuesta 3', button_style='info')
    button3.on_click(verificar_pregunta3)

    print("\nIngresa tu(s) respuesta(s):")
    display(HBox([Label('Primera posición:'), input3a, Label('pies')]))
    display(HBox([Label('Segunda posición (si existe):'), input3b, Label('pies')]))
    display(button3)
    display(output3)

    # Resumen final
    print("\n\n" + "="*60)
    print("INSTRUCCIONES FINALES")
    print("="*60)
    print("""
1. Usa el EXPLORADOR INTERACTIVO (slider arriba) para observar cómo 
   cambia la flexión en diferentes posiciones de la viga.

2. Responde las tres preguntas usando los valores que observas en la 
   gráfica y en los valores numéricos mostrados.

3. Haz clic en los botones "Verificar Respuesta" para comprobar si 
   tus respuestas son correctas.

4. Si una respuesta es incorrecta, vuelve a usar el explorador para 
   encontrar el valor correcto.

¡Buena suerte! 🚀
""")
#if __name__ == "__main__" and "__file__" in globals():
 #   main()