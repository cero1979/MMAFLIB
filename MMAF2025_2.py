# Contenido COMPLETO Y CORRECTO para MMAF2025_2.py

"""
Biblioteca de funciones interactivas para el cuaderno MMAF.
Contiene:
1. Demo: Podar Campo
2. Demo: Drones
3. Demo: División Sintética
4. Demo: Flexión de Viga
"""

# ===================================================================
# 1. IMPORTACIONES GLOBALES
# ===================================================================
from __future__ import annotations
import random
from fractions import Fraction
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import clear_output, display, Markdown, HTML
from ipywidgets import IntSlider, FloatSlider, Button, VBox, HBox, Label, FloatText, Output, interact, interactive

# Configuración de estilo de gráficos
plt.style.use('seaborn-v0_8-darkgrid')


# ===================================================================
# 2. DEMO: PODAR CAMPO
# ===================================================================

def podar_campo_demo(b0: float = 10.0, x0: float = 2.0):
    """Muestra un gráfico interactivo del área podada de un campo cuadrado."""

    out = widgets.Output()
    b_slider = widgets.FloatSlider(value=b0, min=0.1, max=20.0, step=0.1, description="b")
    x_slider = widgets.FloatSlider(value=x0, min=0.0, max=b0 / 2, step=0.1, description="x")

    def draw(b: float, x: float) -> None:
        if x > b / 2:
            x = b / 2
            x_slider.value = x

        inner_side = b - 2 * x
        x_slider.max = b / 2

        fig, ax_field = plt.subplots(figsize=(6, 6))
        ax_field.set_aspect("equal")
        ax_field.axis("off")

        outer_sq = plt.Rectangle((0, 0), b, b, facecolor="#d5f2c2", edgecolor="black")
        inner_sq = plt.Rectangle((x, x), inner_side, inner_side, facecolor="white", edgecolor="black")
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
        ax_field.text(1.05, 1, text, transform=ax_field.transAxes, va="top", ha="left")

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


# ===================================================================
# 3. DEMO: DRONES
# ===================================================================

def _generar_escenario_drones():
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


def _evitar_colision_factory_drones():
    """Crear una función interactiva para evitar la colisión de drones."""
    C_real, D1, D2, v1, v2, S1, S2 = _generar_escenario_drones()

    def evitar_colision(x_c, y_c):
        C = np.array([x_c, y_c])
        distancia = np.linalg.norm(C - C_real)
        tolerancia = 0.05
        t_vals = np.linspace(-0.2, 1.4, 200)
        l1_x, l1_y = D1[0] + t_vals * v1[0], D1[1] + t_vals * v1[1]
        l2_x, l2_y = D2[0] + t_vals * v2[0], D2[1] + t_vals * v2[1]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(l1_x, l1_y, color="dodgerblue", linewidth=2, label="Trayectoria dron 1")
        ax.plot(l2_x, l2_y, color="darkorange", linewidth=2, label="Trayectoria dron 2")
        ax.scatter(*D1, color="dodgerblue", s=80, label="Dron 1")
        ax.scatter(*D2, color="darkorange", s=80, label="Dron 2")
        ax.scatter(*C, color="purple", marker="*", s=200, label=fr"Punto propuesto C({x_c}, {y_c})")
        ax.scatter(*C_real, color="black", marker="x", s=100, zorder=10, label="Punto de cruce real")

        if distancia <= tolerancia:
            ax.scatter(*S1, color="navy", marker="s", s=120, label=fr"S₁({S1[0]:.2f}, {S1[1]:.2f})")
            ax.scatter(*S2, color="sienna", marker="s", s=120, label=fr"S₂({S2[0]:.2f}, {S2[1]:.2f})")

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
            print(f"S₁ = ({S1[0]:.2f}, {S1[1]:.2f})")
            print(f"S₂ = ({S2[0]:.2f}, {S2[1]:.2f})")
        else:
            print("Ajusta C: los puntos extra S₁ y S₂ aparecerán cuando aciertes el cruce real.")
    return evitar_colision


def lanzar_demo_drones():
    """Mostrar los controles interactivos del ejercicio de drones."""
    print("=== DEMO: CRUCE DE DRONES ===")
    return interact(
        _evitar_colision_factory_drones(),
        x_c=IntSlider(value=0, min=-2, max=5, step=1, description="x de C"),
        y_c=IntSlider(value=0, min=0, max=5, step=1, description="y de C"),
    )

# ===================================================================
# 4. DEMO: DIVISIÓN SINTÉTICA
# ===================================================================

def _formatear_fraccion(valor: Fraction | None) -> str:
    if valor is None: return ""
    if isinstance(valor, Fraction):
        if valor.denominator == 1: return str(valor.numerator)
        return f"{valor.numerator}/{valor.denominator}"
    return str(valor)


def _parsear_coeficientes(texto: str, grado: int) -> List[Fraction]:
    partes = texto.replace(",", " ").split()
    if len(partes) != grado + 1:
        raise ValueError(f"Se esperaban {grado + 1} coeficientes y se recibieron {len(partes)}.")
    try:
        return [Fraction(parte) for parte in partes]
    except ValueError as exc:
        raise ValueError("Coeficientes no válidos. Usa enteros, fracciones (a/b) o decimales.") from exc


def _division_sintetica_algoritmo(
    coeficientes: Sequence[Fraction], r: Fraction
) -> Tuple[List[Fraction], List[Fraction | None], List[Fraction]]:
    productos: List[Fraction | None] = [None] * len(coeficientes)
    acumulados: List[Fraction] = [Fraction(0)] * len(coeficientes)
    for indice, coef in enumerate(coeficientes):
        suma = coef + (productos[indice] or 0)
        acumulados[indice] = suma
        if indice < len(coeficientes) - 1:
            productos[indice + 1] = suma * r
    return list(coeficientes), productos, acumulados


def _tabla_html_division(
    encabezado: Iterable[str], filas: Iterable[Iterable[str]],
) -> str:
    def celda(e: str, tag: str = "td") -> str:
        return f"<{tag} style='padding:6px 10px;text-align:center;border:1px solid #bbb;'>{e}</{tag}>"
    filas_html = ["<tr>" + "".join(celda(v) for v in fila) + "</tr>" for fila in filas]
    enc_html = "<tr>" + "".join(celda(v, "th") for v in encabezado) + "</tr>"
    return (
        f"<table style='border-collapse:collapse;font-family:Segoe UI, sans-serif;margin-top:10px;'>"
        f"{enc_html}{''.join(filas_html)}</table>"
    )


def _construir_tabla_division(
    coefs: Sequence[Fraction], prods: Sequence[Fraction | None], acums: Sequence[Fraction],
) -> str:
    grados = [f"x^{i}" for i in range(len(coefs) - 1, -1, -1)]
    encabezado = ["Columna"] + grados
    fila_coef = ["Coeficientes"] + [_formatear_fraccion(c) for c in coefs]
    fila_prod = ["Multiplicaciones"] + [_formatear_fraccion(p) for p in prods]
    fila_acum = ["Suma parcial"] + [_formatear_fraccion(a) for a in acums]
    return _tabla_html_division(encabezado, [fila_coef, fila_prod, fila_acum])


def _describir_resultado_division(acumulados: Sequence[Fraction]) -> str:
    if not acumulados: return "No hay datos."
    resto = acumulados[-1]
    cociente = acumulados[:-1]
    texto_cociente = ", ".join(_formatear_fraccion(c) for c in cociente) or "0"
    return (
        f"**Coeficientes del cociente:** {texto_cociente}\n\n"
        f"**Resto:** {_formatear_fraccion(resto)}"
    )


def crear_interfaz_division_sintetica() -> None:
    """Construye y muestra los controles interactivos para la división sintética."""
    print("=== DEMO: DIVISIÓN SINTÉTICA ===")
    estilo = {"description_width": "initial"}
    grado_input = widgets.BoundedIntText(value=3, min=0, max=20, description="Grado del polinomio", style=estilo)
    coef_input = widgets.Textarea(value="1 -16 -60 432", description="Coeficientes (de x^n a x^0)", layout=widgets.Layout(width="100%", height="80px"), style=estilo)
    r_input = widgets.Text(value="4", description="Valor de r (x - r)", style=estilo)
    boton = widgets.Button(description="Calcular tabla", button_style="success", icon="calculator")
    salida = widgets.Output()

    def _calcular(_=None) -> None:
        with salida:
            salida.clear_output()
            try:
                grado = grado_input.value
                coeficientes = _parsear_coeficientes(coef_input.value, grado)
                r = Fraction(r_input.value)
            except ValueError as error:
                display(Markdown(f"**Error:** {error}"))
                return
            coef, prods, acums = _division_sintetica_algoritmo(coeficientes, r)
            tabla_html = _construir_tabla_division(coef, prods, acums)
            resumen = _describir_resultado_division(acums)
            display(Markdown("**Tabla de división sintética:**"))
            display(HTML(tabla_html))
            display(Markdown(resumen))

    boton.on_click(_calcular)
    display(
        widgets.VBox([
            widgets.HTML("<h3>División sintética interactiva</h3>"),
            widgets.HTML("<p>Introduce el grado, coeficientes de <em>P(x)</em> y <em>r</em> del divisor <strong>(x - r)</strong>."),
            grado_input, coef_input, r_input, boton, salida,
        ])
    )
    _calcular()


# ===================================================================
# 5. DEMO: FLEXIÓN DE VIGA
# ===================================================================

def _deflection_viga(x):
    return (1/16000) * (60 * x**2 - x**3)


def _plot_beam_deflection(x_position=10.0):
    x = np.linspace(0, 20, 500)
    y = _deflection_viga(x)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    ax1.plot(x, y, 'b-', linewidth=2.5, label='Curva de flexión d(x)')
    ax1.axvline(x_position, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'x = {x_position:.2f} pies')
    ax1.plot(x_position, _deflection_viga(x_position), 'ro', markersize=12, label=f'd = {_deflection_viga(x_position):.4f} pulg')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Posición x (pies)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Flexión d (pulgadas)', fontsize=12, fontweight='bold')
    ax1.set_title('Flexión de la Viga en Voladizo', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 20)
    
    ax2.set_xlim(-2, 22); ax2.set_ylim(-3, 2); ax2.axis('off')
    viga_y = -_deflection_viga(x) * 0.3
    ax2.plot(x, viga_y, 'brown', linewidth=8, solid_capstyle='round')
    ax2.fill_between([-1, 0], [-2, -2], [2, 2], color='gray', alpha=0.5, hatch='///')
    ax2.plot([0, 0], [-2, 2], 'k-', linewidth=3)
    ax2.arrow(20, -_deflection_viga(20)*0.3, 0, -1.5, head_width=0.5, head_length=0.3, fc='darkred', ec='darkred', linewidth=2)
    ax2.text(20, -_deflection_viga(20)*0.3 - 2, '600 lb', ha='center', fontsize=12, fontweight='bold', color='darkred')
    ax2.plot(x_position, -_deflection_viga(x_position)*0.3, 'ro', markersize=15, zorder=5)
    ax2.arrow(x_position, 1.5, 0, -1.3 + _deflection_viga(x_position)*0.3, head_width=0.3, head_length=0.2, fc='red', ec='red', linewidth=1.5)
    ax2.text(x_position, 1.8, f'x = {x_position:.1f} ft', ha='center', fontsize=11, fontweight='bold', color='red')
    ax2.plot([0, 20], [-2.5, -2.5], 'k-', linewidth=1); ax2.plot([0, 0], [-2.5, -2.7], 'k-', linewidth=1)
    ax2.plot([20, 20], [-2.5, -2.7], 'k-', linewidth=1); ax2.text(10, -2.9, '20 pies', ha='center', fontsize=11, fontweight='bold')
    ax2.set_title('Diagrama Esquemático', fontsize=14, fontweight='bold')
    
    plt.tight_layout(); plt.show()
    print(f"\n📍 Posición seleccionada: x = {x_position:.3f} pies")
    print(f"📏 Flexión calculada: d = {_deflection_viga(x_position):.6f} pulgadas")


def lanzar_demo_viga():
    """Crea y muestra la interfaz completa de la demo de flexión de viga."""
    
    respuestas = {'p1': None, 'p2': None, 'p3a': None, 'p3b': None}

    def verificar_pregunta1(b):
        with output1:
            clear_output()
            x_resp, d_resp = input1_x.value, input1_d.value
            x_correcto, d_correcto = 20.0, _deflection_viga(20.0)
            if abs(x_resp - x_correcto) < 0.5 and abs(d_resp - d_correcto) < 0.05:
                print(f"✅ ¡CORRECTO! x = {x_correcto} pies, d_max = {d_correcto:.4f} pulgadas")
                respuestas['p1'] = True
            else:
                print("❌ No es correcto. Usa el explorador para encontrar el punto más alto.")

    def verificar_pregunta2(b):
        with output2:
            clear_output()
            d_resp, d_correcto = input2.value, _deflection_viga(10.0)
            if abs(d_resp - d_correcto) < 0.01:
                print(f"✅ ¡CORRECTO! d = {d_correcto:.4f} pulgadas")
                respuestas['p2'] = True
            else:
                print("❌ No es correcto. Posiciona el slider en x = 10.")

    def verificar_pregunta3(b):
        with output3:
            clear_output()
            sols_correctas = [7.54] # Solución numérica de x³ - 60x² + 8000 = 0 en [0, 20]
            resp_usuario = [r for r in [input3a.value, input3b.value] if r > 0]
            correctas = 0
            for r_usr in resp_usuario:
                if any(abs(r_usr - r_sol) < 0.3 for r_sol in sols_correctas):
                    correctas += 1
            if correctas == len(sols_correctas):
                print(f"✅ ¡CORRECTO! La posición es x ≈ {sols_correctas[0]:.2f} pies.")
                respuestas['p3a'] = True
            else:
                print("❌ No es correcto. Mueve el slider lentamente y busca d ≈ 0.5")

    print("="*60); print("PROBLEMA: FLEXIÓN DE UNA VIGA EN VOLADIZO"); print("="*60)
    print("\nd(x) = (1/16000)(60x² - x³), para 0 ≤ x ≤ 20\n")
    print("="*60); print("EXPLORADOR INTERACTIVO"); print("="*60)

    slider_widget = interactive(_plot_beam_deflection, x_position=FloatSlider(min=0, max=20, step=0.1, value=10, description='Posición x:', style={'description_width': '100px'}))
    display(slider_widget)

    print("\n\n" + "="*60); print("PREGUNTAS A RESOLVER"); print("="*60)

    print("\n📌 PREGUNTA 1: Flexión máxima"); print("-" * 60)
    output1 = Output(); input1_x = FloatText(description='x =', style={'description_width': '50px'}, step=0.01)
    input1_d = FloatText(description='d_max =', style={'description_width': '50px'}, step=0.001)
    button1 = Button(description='Verificar Respuesta 1', button_style='info'); button1.on_click(verificar_pregunta1)
    display(HBox([Label('Posición:'), input1_x, Label('pies')]), HBox([Label('Flexión:'), input1_d, Label('pulgadas')]), button1, output1)

    print("\n\n📌 PREGUNTA 2: Flexión en la mitad (x = 10)"); print("-" * 60)
    output2 = Output(); input2 = FloatText(description='d(10) =', style={'description_width': '60px'}, step=0.001)
    button2 = Button(description='Verificar Respuesta 2', button_style='info'); button2.on_click(verificar_pregunta2)
    display(HBox([input2, Label('pulgadas')]), button2, output2)

    print("\n\n📌 PREGUNTA 3: Posición para flexión = 0.5 pulg"); print("-" * 60)
    output3 = Output(); input3a = FloatText(description='x₁ =', style={'description_width': '50px'}, step=0.01)
    input3b = FloatText(description='x₂ =', style={'description_width': '50px'}, step=0.01)
    button3 = Button(description='Verificar Respuesta 3', button_style='info'); button3.on_click(verificar_pregunta3)
    display(HBox([input3a, input3b, Label('pies')]), button3, output3)
    
    print("\n\n" + "="*60); print("INSTRUCCIONES FINALES"); print("="*60)
    print("Usa el explorador interactivo y los botones para verificar tus respuestas. ¡Suerte! 🚀")


# --- FIN DEL ARCHIVO ---
Paso 2: Usa esta celda de carga en tu Notebook
Este es el paso CRÍTICO. El error NameError ocurre porque exec no está poniendo las funciones en la memoria "global" del cuaderno.

Usa exactamente este código en tu celda de carga. La parte exec(response.text, globals()) es la clave que soluciona el NameError.

Python

# Celda 1: Cargar todas las funciones (VERSIÓN DEFINITIVA)
import requests
import time

# (1) Forzar la descarga de la versión más reciente (evita el caché)
timestamp = int(time.time())
url = f"https://raw.githubusercontent.com/cero1979/MMAFLIB/main/MMAF2025_2.py?t={timestamp}"

print(f"Descargando la biblioteca desde: {url}")

try:
    response = requests.get(url)
    response.raise_for_status() 
    
    # (2) Ejecutar el código en el ÁMBITO GLOBAL (globals())
    # ESTO ES LO QUE SOLUCIONA EL 'NameError'
    exec(response.text, globals())
    
    print("✅ Biblioteca cargada. Funciones disponibles.")
    print("Funciones de demo disponibles:")
    print(" - podar_campo_demo()")
    print(" - lanzar_demo_drones()")
    print(" - crear_interfaz_division_sintetica()")
    print(" - lanzar_demo_viga()")

except Exception as e:
    print(f"❌ ERROR al cargar el script: {e}")