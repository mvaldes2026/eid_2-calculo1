from dataclasses import dataclass
import os
import sys
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt

# Modo consola por defecto (ideal para la evaluación); usa --gui para abrir el simulador visual
USAR_CLI_FORZADO = "--gui" not in sys.argv
if USAR_CLI_FORZADO or (os.name != "nt" and not os.environ.get("DISPLAY")):
    matplotlib.use("Agg")


# ==============================================================================
# 1. CONFIGURACIÓN DEL EQUIPO Y PARÁMETROS
# ==============================================================================
RUT_POR_DEFECTO = "22.549.846-6"
INTEGRANTES = ["Mauricio Valdés", "Joaquín Navarrete"]
C_MAX = 10.0


@dataclass(frozen=True)
class ModeloCaso:
    codigo: str
    nombre: str
    formula_texto: str
    funcion_termino: Callable[[int, float], float]


@dataclass(frozen=True)
class ResultadoCombinacion:
    caso: str
    nombre_p: str
    p: float
    converge: bool
    limite_empirico: float
    iteraciones_conv: int
    cumple_umbral: bool


# ==============================================================================
# 2. DEFINICIÓN MODULAR DE CASOS (TRABAJO POR RAMAS)
# ==============================================================================

# --- RAMA: feature/caso-a (IMPLEMENTADO POR JOAQUÍN) ---
def termino_caso_a(i: int, p: float) -> float:
    """Caso A (WiFi estándar): i * (1 - p)^(i - 1) * p"""
    return float(i) * ((1.0 - p) ** (i - 1)) * p


# --- RAMA: feature/casos-bc (IMPLEMENTADO POR MAURICIO) ---
def termino_caso_b(i: int, p: float) -> float:
    """Caso B (Fibra con reintentos decrecientes): (1 / i) * (1 - p)^(i - 1) * p"""
    return (1.0 / i) * ((1.0 - p) ** (i - 1)) * p


def termino_caso_c(i: int, p: float) -> float:
    """Caso C (Red móvil con copia extra): (i + 1) * (1 - p)^(i - 1) * p"""
    return (i + 1) * ((1.0 - p) ** (i - 1)) * p


MODELOS_ARQ: dict[str, ModeloCaso] = {
    "A": ModeloCaso(
        codigo="A",
        nombre="WiFi estándar",
        formula_texto="S_k^A = sum_{i=1}^k i * (1 - p)^(i - 1) * p",
        funcion_termino=termino_caso_a,
    ),
    "B": ModeloCaso(
        codigo="B",
        nombre="Fibra con reintentos decrecientes",
        formula_texto="S_k^B = sum_{i=1}^k (1/i) * (1 - p)^(i - 1) * p",
        funcion_termino=termino_caso_b,
    ),
    "C": ModeloCaso(
        codigo="C",
        nombre="Red móvil con copia extra",
        formula_texto="S_k^C = sum_{i=1}^k (i + 1) * (1 - p)^(i - 1) * p",
        funcion_termino=termino_caso_c,
    ),
}


# ==============================================================================
# 3. VALIDACIÓN DE RUT Y CÁLCULOS MATEMÁTICOS
# ==============================================================================
def extraer_digitos_rut(rut_raw: str) -> tuple[str, list[int]]:
    if not isinstance(rut_raw, str) or not rut_raw.strip():
        raise ValueError("El RUT no puede estar vacío.")

    cuerpo = rut_raw.strip().split("-")[0]
    solo_digitos = "".join(ch for ch in cuerpo if ch.isdigit())
    if len(solo_digitos) < 8:
        raise ValueError(f"RUT inválido ('{rut_raw}'): se requieren 8 dígitos antes del guion.")

    ocho_digitos = solo_digitos[:8]
    return ocho_digitos, [int(d) for d in ocho_digitos]


def calcular_probabilidades(digitos: list[int]) -> dict[str, float]:
    d1, d2, d3, d4, d5, d6, d7, d8 = digitos
    p1 = ((d1 + d3 + d5 + 1) * 2) / 100.0
    p2 = ((d2 + d4 + d6 + 1) * 3) / 100.0
    p3 = ((d7 + d8 + 1) * 4) / 100.0

    probs = {"p1": p1, "p2": p2, "p3": p3}
    for nombre, valor in probs.items():
        if not (0.0 < valor < 1.0):
            raise ValueError(f"Probabilidad fuera del intervalo (0, 1): {nombre} = {valor}")
    return probs


def caso_esta_implementado(caso: str) -> bool:
    try:
        MODELOS_ARQ[caso].funcion_termino(1, 0.5)
        return True
    except NotImplementedError:
        return False


def calcular_sumas_parciales(caso: str, p: float, k_max: int) -> list[float]:
    fn = MODELOS_ARQ[caso].funcion_termino
    sumas: list[float] = []
    acumulado = 0.0
    for i in range(1, k_max + 1):
        acumulado += fn(i, p)
        sumas.append(acumulado)
    return sumas


def evaluar_convergencia(
    caso: str, p: float, tol: float = 1e-12, max_iter: int = 100_000
) -> tuple[bool, float, int]:
    fn = MODELOS_ARQ[caso].funcion_termino
    acumulado = 0.0
    min_pasos = int(2.0 / p) + 10

    for i in range(1, max_iter + 1):
        termino = fn(i, p)
        acumulado += termino
        if i >= min_pasos and abs(termino) < tol:
            return True, acumulado, i
    return False, acumulado, max_iter


def generar_texto_recomendacion(resultados: list[ResultadoCombinacion], c_max: float = C_MAX) -> str:
    viables = [r for r in resultados if r.cumple_umbral]
    if not viables:
        return (
            f"Ninguna combinación cumple con el umbral máximo C_max = {c_max:.1f} "
            "transmisiones esperadas."
        )

    mejor = min(viables, key=lambda r: (r.limite_empirico, r.iteraciones_conv))
    texto = (
        f"Se recomienda Caso {mejor.caso} ({MODELOS_ARQ[mejor.caso].nombre}) con "
        f"{mejor.nombre_p} = {mejor.p:.2f}: la serie converge monotónicamente a un límite "
        f"empírico de {mejor.limite_empirico:.6f} <= {c_max:.1f} transmisiones esperadas, "
        f"estabilizándose (error < 1e-12) en k = {mejor.iteraciones_conv} iteraciones y "
        f"minimizando el costo esperado."
    )

    viables_a = [r for r in viables if r.caso == "A"]
    if viables_a and mejor.caso != "A":
        mejor_a = min(viables_a, key=lambda r: (r.limite_empirico, r.iteraciones_conv))
        texto += (
            f"\nAlternativa WiFi (Caso A): Con {mejor_a.nombre_p} = {mejor_a.p:.2f} "
            f"converge a {mejor_a.limite_empirico:.6f} (k = {mejor_a.iteraciones_conv})."
        )
    return texto


def ejecutar_analisis(rut_raw: str, c_max: float = C_MAX) -> dict:
    rut_limpio, digitos = extraer_digitos_rut(rut_raw)
    probs = calcular_probabilidades(digitos)
    casos_disponibles = [c for c in sorted(MODELOS_ARQ.keys()) if caso_esta_implementado(c)]

    tablas_s10: dict[str, dict[str, list[float]]] = {}
    resultados: list[ResultadoCombinacion] = []

    for caso in casos_disponibles:
        tablas_s10[caso] = {
            nombre_p: calcular_sumas_parciales(caso, p_val, 10)
            for nombre_p, p_val in probs.items()
        }
        for nombre_p, p_val in probs.items():
            conv, lim_emp, iters = evaluar_convergencia(caso, p_val)
            resultados.append(
                ResultadoCombinacion(
                    caso=caso,
                    nombre_p=nombre_p,
                    p=p_val,
                    converge=conv,
                    limite_empirico=lim_emp,
                    iteraciones_conv=iters,
                    cumple_umbral=conv and (lim_emp <= c_max),
                )
            )

    return {
        "rut_ingresado": rut_raw,
        "rut_8_digitos": rut_limpio,
        "digitos": digitos,
        "probabilidades": probs,
        "c_max": c_max,
        "casos_disponibles": casos_disponibles,
        "tablas_s10": tablas_s10,
        "resultados": resultados,
        "recomendacion": generar_texto_recomendacion(resultados, c_max),
    }


# ==============================================================================
# 4. IMPRESIÓN EN CONSOLA
# ==============================================================================
def imprimir_reporte(datos: dict) -> None:
    probs = datos["probabilidades"]
    c_max = datos["c_max"]
    col_p1 = f"p1={probs['p1']:.2f}"
    col_p2 = f"p2={probs['p2']:.2f}"
    col_p3 = f"p3={probs['p3']:.2f}"

    print("=" * 82)
    print("EID N°2 - MATE1133: ANÁLISIS DE CONVERGENCIA PROTOCOLO ARQ")
    print("=" * 82)
    print(f"Integrantes : {', '.join(INTEGRANTES)}")
    print(f"RUT elegido : {datos['rut_ingresado']} -> Dígitos = {datos['digitos']}")
    print(f"Parámetros  : p1 = {probs['p1']:.4f} | p2 = {probs['p2']:.4f} | p3 = {probs['p3']:.4f}")
    print("=" * 82)

    for caso in sorted(MODELOS_ARQ.keys()):
        modelo = MODELOS_ARQ[caso]
        if caso not in datos["casos_disponibles"]:
            print(f"\n--- CASO {caso}: {modelo.nombre} [PENDIENTE DE IMPLEMENTACIÓN] ---")
            continue

        print(f"\n--- CASO {caso}: {modelo.nombre} ({modelo.formula_texto}) ---")
        print(f"{'k':>4} | {col_p1:>14} | {col_p2:>14} | {col_p3:>14}")
        print("-" * 55)
        series_p = datos["tablas_s10"][caso]
        for idx in range(10):
            print(
                f"{idx + 1:>4} | {series_p['p1'][idx]:>14.6f} | "
                f"{series_p['p2'][idx]:>14.6f} | {series_p['p3'][idx]:>14.6f}"
            )

    print("\n" + "=" * 82)
    print(f"CONVERGENCIA Y TABLA DE VIABILIDAD (UMBRAL C_max = {c_max:.1f})")
    print("=" * 82)
    print(
        f"{'Caso':<6} | {'Probabilidad':<13} | {'¿Converge?':<11} | "
        f"{'Límite Empírico':>16} | {'Viabilidad':<12}"
    )
    print("-" * 82)
    for r in datos["resultados"]:
        estado_conv = "Sí" if r.converge else "No"
        estado_viab = "CUMPLE" if r.cumple_umbral else "NO CUMPLE"
        print(
            f"{r.caso:<6} | {r.nombre_p} ({r.p:.2f})     | {estado_conv:<11} | "
            f"{r.limite_empirico:>16.6f} | {estado_viab:<12}"
        )

    print("-" * 82)
    print("RECOMENDACIÓN:")
    print(datos["recomendacion"])
    print("=" * 82)


# ==============================================================================
# 5. GENERACIÓN DE GRÁFICOS MATPLOTLIB (INDIVIDUALES + COMPARATIVOS)
# ==============================================================================
def crear_figura_caso(caso: str, datos: dict, k_override: int | None = None) -> plt.Figure:
    modelo = MODELOS_ARQ[caso]
    res_caso = [r for r in datos["resultados"] if r.caso == caso]
    colores = {"p1": "#2563eb", "p2": "#ea580c", "p3": "#16a34a"}

    if k_override is not None:
        k_max_plot = max(10, int(k_override))
    else:
        k_max_plot = min(max(max(r.iteraciones_conv for r in res_caso) // 2, 25), 90)
    eje_k = list(range(1, k_max_plot + 1))

    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=110)
    fig.patch.set_facecolor("#ffffff")
    for r in res_caso:
        color = colores[r.nombre_p]
        y_vals = calcular_sumas_parciales(caso, r.p, k_max_plot)
        ax.plot(
            eje_k,
            y_vals,
            marker="o",
            markersize=3,
            linewidth=2.0,
            color=color,
            label=f"{r.nombre_p} = {r.p:.2f} (Límite ≈ {r.limite_empirico:.4f})",
        )
        ax.axhline(y=r.limite_empirico, color=color, linestyle="--", alpha=0.65, linewidth=1.3)

    ax.set_title(f"Caso {caso}: {modelo.nombre}\n{modelo.formula_texto}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Número de términos acumulados (k)", fontsize=10)
    ax.set_ylabel("Costo / Transmisiones esperadas (S_k)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def crear_figura_resumen_barras(datos: dict) -> plt.Figure:
    resultados = datos["resultados"]
    c_max = datos["c_max"]
    etiquetas = [f"{r.caso}-{r.nombre_p}\n({r.p:.2f})" for r in resultados]
    valores = [r.limite_empirico for r in resultados]
    colores = ["#16a34a" if r.cumple_umbral else "#dc2626" for r in resultados]

    fig, ax = plt.subplots(figsize=(7.8, 2.7), dpi=105)
    fig.patch.set_facecolor("#ffffff")
    barras = ax.bar(etiquetas, valores, color=colores, width=0.55, edgecolor="#1e293b", linewidth=0.7)
    ax.axhline(
        y=c_max,
        color="#dc2626",
        linestyle="--",
        linewidth=1.6,
        label=f"Umbral máximo C_max = {c_max:.1f}",
    )

    for barra, val in zip(barras, valores):
        ax.text(
            barra.get_x() + barra.get_width() / 2.0,
            val + (max(max(valores), c_max) * 0.02),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#0f172a",
        )

    ax.set_ylim(0, max(max(valores), c_max) * 1.2)
    ax.set_title("Comparación de Límites Empíricos vs Umbral Operativo (C_max)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Límite S_∞", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def crear_figura_comparativa_prob(datos: dict, nombre_p: str, k_max_plot: int) -> plt.Figure:
    p_val = datos["probabilidades"][nombre_p]
    eje_k = list(range(1, k_max_plot + 1))
    colores_caso = {"A": "#2563eb", "B": "#16a34a", "C": "#9333ea"}

    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=110)
    fig.patch.set_facecolor("#ffffff")

    for caso in datos["casos_disponibles"]:
        modelo = MODELOS_ARQ[caso]
        res = next(r for r in datos["resultados"] if r.caso == caso and r.nombre_p == nombre_p)
        y_vals = calcular_sumas_parciales(caso, p_val, k_max_plot)
        color = colores_caso[caso]
        ax.plot(
            eje_k,
            y_vals,
            marker="o",
            markersize=3,
            linewidth=2.1,
            color=color,
            label=f"Caso {caso} ({modelo.nombre}) → Límite ≈ {res.limite_empirico:.4f}",
        )
        ax.axhline(y=res.limite_empirico, color=color, linestyle="--", alpha=0.6, linewidth=1.2)

    ax.set_title(
        f"Comparativa Directa entre Arquitecturas para {nombre_p} = {p_val:.2f}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Número de términos acumulados (k)", fontsize=10)
    ax.set_ylabel("Costo / Transmisiones esperadas (S_k)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def guardar_graficos_png(datos: dict) -> dict[str, plt.Figure]:
    figuras: dict[str, plt.Figure] = {}
    for caso in datos["casos_disponibles"]:
        fig = crear_figura_caso(caso, datos)
        fig.savefig(f"grafico_caso_{caso}.png", dpi=150)
        figuras[caso] = fig
    return figuras


# ==============================================================================
# 6. APP INTERACTIVA DE PROYECCIÓN (TKINTER + MATPLOTLIB EMBEBIDO)
# ==============================================================================
def lanzar_interfaz_grafica(datos_iniciales: dict, figuras_iniciales: dict[str, plt.Figure]) -> bool:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    except Exception:
        return False

    try:
        root = tk.Tk()
    except Exception:
        return False

    root.title("EID N°2 - MATE1133 | Simulador Interactivo Protocolo ARQ")
    root.geometry("1240x800")
    root.minsize(1040, 700)
    root.configure(bg="#f1f5f9")

    estilo = ttk.Style(root)
    if "clam" in estilo.theme_names():
        estilo.theme_use("clam")

    estilo.configure("TFrame", background="#f1f5f9")
    estilo.configure("TNotebook", background="#f1f5f9", borderwidth=0)
    estilo.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(14, 7))
    estilo.map(
        "TNotebook.Tab",
        background=[("selected", "#ffffff"), ("!selected", "#e2e8f0")],
        foreground=[("selected", "#0f172a"), ("!selected", "#475569")],
    )
    estilo.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background="#e2e8f0", foreground="#0f172a")
    estilo.configure("Treeview", rowheight=27, font=("Consolas", 10), background="#ffffff", fieldbackground="#ffffff")

    figuras_activas: list[plt.Figure] = list(figuras_iniciales.values())

    def limpiar_figuras() -> None:
        for f in figuras_activas:
            plt.close(f)
        figuras_activas.clear()

    # --------------------------------------------------------------------------
    # 1. BARRA SUPERIOR TIPO APP DE PRESENTACIÓN
    # --------------------------------------------------------------------------
    header = tk.Frame(root, bg="#0f172a", padx=18, pady=12)
    header.pack(fill=tk.X)

    header_izq = tk.Frame(header, bg="#0f172a")
    header_izq.pack(side=tk.LEFT)

    tk.Label(
        header_izq,
        text="PROTOCOLO ARQ  |  SIMULADOR DE CONVERGENCIA Y VIABILIDAD",
        font=("Segoe UI", 14, "bold"),
        fg="#f8fafc",
        bg="#0f172a",
    ).pack(anchor=tk.W)

    tk.Label(
        header_izq,
        text="Cálculo Básico (MATE1133) • Evaluación Integral de Desempeño N°2 • Series e Iteraciones",
        font=("Segoe UI", 10),
        fg="#94a3b8",
        bg="#0f172a",
    ).pack(anchor=tk.W, pady=(2, 0))

    header_der = tk.Frame(header, bg="#1e293b", padx=12, pady=6)
    header_der.pack(side=tk.RIGHT)

    tk.Label(
        header_der,
        text="EQUIPO DE TRABAJO",
        font=("Segoe UI", 8, "bold"),
        fg="#38bdf8",
        bg="#1e293b",
    ).pack(anchor=tk.E)

    tk.Label(
        header_der,
        text="  •  ".join(INTEGRANTES),
        font=("Segoe UI", 10, "bold"),
        fg="#f1f5f9",
        bg="#1e293b",
    ).pack(anchor=tk.E)

    # --------------------------------------------------------------------------
    # 2. BARRA DE CONTROLES INTERACTIVOS (RUT, UMBRAL C_max, HORIZONTE k)
    # --------------------------------------------------------------------------
    panel_ctrl = tk.Frame(root, bg="#ffffff", padx=16, pady=10, highlightbackground="#cbd5e1", highlightthickness=1)
    panel_ctrl.pack(fill=tk.X, padx=14, pady=(12, 8))

    tk.Label(panel_ctrl, text="RUT:", font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#0f172a").pack(side=tk.LEFT)
    rut_var = tk.StringVar(master=root, value=datos_iniciales["rut_ingresado"])
    entry_rut = tk.Entry(
        panel_ctrl,
        textvariable=rut_var,
        width=15,
        font=("Consolas", 11, "bold"),
        bg="#f8fafc",
        fg="#0f172a",
        relief="solid",
        bd=1,
    )
    entry_rut.pack(side=tk.LEFT, padx=(6, 14), ipady=3)

    tk.Label(panel_ctrl, text="Umbral C_max:", font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#0f172a").pack(side=tk.LEFT)
    cmax_var = tk.StringVar(master=root, value=f"{datos_iniciales['c_max']:.1f}")
    entry_cmax = tk.Entry(
        panel_ctrl,
        textvariable=cmax_var,
        width=7,
        font=("Consolas", 11, "bold"),
        bg="#f8fafc",
        fg="#0f172a",
        relief="solid",
        bd=1,
        justify=tk.CENTER,
    )
    entry_cmax.pack(side=tk.LEFT, padx=(6, 14), ipady=3)

    tk.Label(panel_ctrl, text="Horizonte gráfico (k):", font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#0f172a").pack(side=tk.LEFT)
    k_slider_var = tk.IntVar(master=root, value=50)
    scale_k = tk.Scale(
        panel_ctrl,
        from_=15,
        to=100,
        orient=tk.HORIZONTAL,
        variable=k_slider_var,
        length=140,
        bg="#ffffff",
        fg="#0f172a",
        highlightthickness=0,
        font=("Consolas", 9),
    )
    scale_k.pack(side=tk.LEFT, padx=(4, 14))

    estado_var = tk.StringVar(master=root, value="Listo • PNGs exportados")

    # --------------------------------------------------------------------------
    # 3. TARJETAS KPI SUPERIORES (CON master=root EXPLÍCITO)
    # --------------------------------------------------------------------------
    kpi_container = tk.Frame(root, bg="#f1f5f9")
    kpi_container.pack(fill=tk.X, padx=14, pady=(0, 8))
    for col_idx in range(4):
        kpi_container.columnconfigure(col_idx, weight=1)

    def crear_tarjeta_kpi(parent: tk.Frame, col: int, titulo: str, acento: str) -> tuple[tk.StringVar, tk.StringVar]:
        card = tk.Frame(parent, bg="#ffffff", highlightbackground="#cbd5e1", highlightthickness=1)
        card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 6, 0 if col == 3 else 6))

        franja = tk.Frame(card, bg=acento, height=4)
        franja.pack(fill=tk.X)

        cuerpo = tk.Frame(card, bg="#ffffff", padx=12, pady=8)
        cuerpo.pack(fill=tk.BOTH, expand=True)

        tk.Label(cuerpo, text=titulo, font=("Segoe UI", 9, "bold"), fg="#64748b", bg="#ffffff").pack(anchor=tk.W)
        val_var = tk.StringVar(master=root)
        sub_var = tk.StringVar(master=root)
        tk.Label(cuerpo, textvariable=val_var, font=("Segoe UI", 13, "bold"), fg="#0f172a", bg="#ffffff").pack(anchor=tk.W, pady=(2, 1))
        tk.Label(cuerpo, textvariable=sub_var, font=("Consolas", 9), fg="#475569", bg="#ffffff").pack(anchor=tk.W)
        return val_var, sub_var

    kpi1_val, kpi1_sub = crear_tarjeta_kpi(kpi_container, 0, "PARÁMETROS DEL RUT", "#2563eb")
    kpi2_val, kpi2_sub = crear_tarjeta_kpi(kpi_container, 1, "VIABILIDAD OPERATIVA", "#16a34a")
    kpi3_val, kpi3_sub = crear_tarjeta_kpi(kpi_container, 2, "MEJOR ESCENARIO GLOBAL", "#0d9488")
    kpi4_val, kpi4_sub = crear_tarjeta_kpi(kpi_container, 3, "MEJOR EN WIFI (CASO A)", "#ea580c")

    def actualizar_kpis(datos: dict) -> None:
        pr = datos["probabilidades"]
        res = datos["resultados"]
        viables = [r for r in res if r.cumple_umbral]

        kpi1_val.set(f"p1={pr['p1']:.2f} | p2={pr['p2']:.2f} | p3={pr['p3']:.2f}")
        kpi1_sub.set(f"Dígitos RUT: {datos['rut_8_digitos']}")

        kpi2_val.set(f"{len(viables)} de {len(res)} CUMPLEN")
        kpi2_sub.set(f"Umbral máximo: C_max <= {datos['c_max']:.1f}")

        if viables:
            mejor = min(viables, key=lambda r: (r.limite_empirico, r.iteraciones_conv))
            kpi3_val.set(f"Caso {mejor.caso} ({mejor.nombre_p}={mejor.p:.2f})")
            kpi3_sub.set(f"Límite: {mejor.limite_empirico:.4f} (k={mejor.iteraciones_conv})")
        else:
            kpi3_val.set("Sin opción viable")
            kpi3_sub.set("Supera umbral C_max")

        viables_a = [r for r in viables if r.caso == "A"]
        if viables_a:
            mejor_a = min(viables_a, key=lambda r: (r.limite_empirico, r.iteraciones_conv))
            kpi4_val.set(f"Caso A ({mejor_a.nombre_p}={mejor_a.p:.2f})")
            kpi4_sub.set(f"Límite: {mejor_a.limite_empirico:.4f} (k={mejor_a.iteraciones_conv})")
        else:
            kpi4_val.set("No viable en Caso A")
            kpi4_sub.set("Excede C_max")

    # --------------------------------------------------------------------------
    # 4. NOTEBOOK CENTRAL CON VISTAS DE PROYECCIÓN
    # --------------------------------------------------------------------------
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 12))

    datos_estado = {"actual": datos_iniciales}

    def renderizar_todo(datos: dict) -> None:
        limpiar_figuras()
        actualizar_kpis(datos)
        k_val = k_slider_var.get()

        pestana_activa = 0
        if notebook.tabs():
            pestana_activa = notebook.index(notebook.select())

        for tab_id in notebook.tabs():
            notebook.forget(tab_id)

        # ======================================================================
        # PESTAÑA 1: TABLERO EJECUTIVO (VIABILIDAD + GRÁFICO BARRAS + RECOMENDACIÓN)
        # ======================================================================
        tab1 = ttk.Frame(notebook, padding=10)
        notebook.add(tab1, text="  1. Resumen Ejecutivo y Viabilidad  ")

        sup_tab1 = tk.Frame(tab1, bg="#f1f5f9")
        sup_tab1.pack(fill=tk.BOTH, expand=True)

        frame_tabla_v = tk.Frame(sup_tab1, bg="#ffffff", padx=10, pady=10, highlightbackground="#cbd5e1", highlightthickness=1)
        frame_tabla_v.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        tk.Label(
            frame_tabla_v,
            text=f"Evaluación de Convergencia y Umbral (C_max = {datos['c_max']:.1f})",
            font=("Segoe UI", 11, "bold"),
            bg="#ffffff",
            fg="#0f172a",
        ).pack(anchor=tk.W, pady=(0, 8))

        cols_v = ("caso", "desc", "prob", "conv", "limite", "iters", "viab")
        tree_v = ttk.Treeview(frame_tabla_v, columns=cols_v, show="headings", height=9)
        encabezados_v = [
            ("caso", "Caso", 65),
            ("desc", "Escenario de Red", 215),
            ("prob", "Probabilidad", 105),
            ("conv", "Converge", 85),
            ("limite", "Límite S_∞", 110),
            ("iters", "k Estab.", 80),
            ("viab", "Estado", 105),
        ]
        for col_id, titulo, ancho in encabezados_v:
            tree_v.heading(col_id, text=titulo)
            tree_v.column(col_id, width=ancho, anchor=tk.W if col_id == "desc" else tk.CENTER)

        tree_v.tag_configure("ok", background="#dcfce7", foreground="#14532d")
        tree_v.tag_configure("fail", background="#fee2e2", foreground="#7f1d1d")

        for r in datos["resultados"]:
            tag = "ok" if r.cumple_umbral else "fail"
            tree_v.insert(
                "",
                tk.END,
                values=(
                    f"Caso {r.caso}",
                    MODELOS_ARQ[r.caso].nombre,
                    f"{r.nombre_p}={r.p:.2f}",
                    "Sí" if r.converge else "No",
                    f"{r.limite_empirico:.6f}",
                    r.iteraciones_conv,
                    "CUMPLE" if r.cumple_umbral else "NO CUMPLE",
                ),
                tags=(tag,),
            )
        tree_v.pack(fill=tk.BOTH, expand=True)

        col_der_t1 = tk.Frame(sup_tab1, bg="#f1f5f9")
        col_der_t1.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        frame_rec = tk.Frame(col_der_t1, bg="#eff6ff", padx=12, pady=10, highlightbackground="#93c5fd", highlightthickness=1)
        frame_rec.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))

        tk.Label(
            frame_rec,
            text="CONCLUSIÓN Y RECOMENDACIÓN DE INGENIERÍA",
            font=("Segoe UI", 10, "bold"),
            bg="#eff6ff",
            fg="#1e3a8a",
        ).pack(anchor=tk.W, pady=(0, 4))

        tk.Label(
            frame_rec,
            text=datos["recomendacion"],
            font=("Segoe UI", 9),
            bg="#eff6ff",
            fg="#0f172a",
            wraplength=540,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X)

        frame_barras = tk.Frame(col_der_t1, bg="#ffffff", padx=8, pady=8, highlightbackground="#cbd5e1", highlightthickness=1)
        frame_barras.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        fig_barras = crear_figura_resumen_barras(datos)
        figuras_activas.append(fig_barras)
        canvas_b = FigureCanvasTkAgg(fig_barras, master=frame_barras)
        canvas_b.draw()
        canvas_b.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ======================================================================
        # PESTAÑAS 2, 3 Y 4: DETALLE POR CADA CASO (TABLA S_1..S_10 + GRÁFICO)
        # ======================================================================
        probs = datos["probabilidades"]
        for idx_tab, caso in enumerate(datos["casos_disponibles"], start=2):
            modelo = MODELOS_ARQ[caso]
            tab_c = ttk.Frame(notebook, padding=10)
            notebook.add(tab_c, text=f"  {idx_tab}. Caso {caso}: {modelo.nombre}  ")

            panel_izq = tk.Frame(tab_c, bg="#ffffff", padx=12, pady=10, highlightbackground="#cbd5e1", highlightthickness=1)
            panel_izq.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

            tk.Label(
                panel_izq,
                text="Sumas Parciales (S_1 a S_10)",
                font=("Segoe UI", 11, "bold"),
                bg="#ffffff",
                fg="#0f172a",
            ).pack(anchor=tk.W, pady=(0, 6))

            cols_s = ("k", "p1", "p2", "p3")
            tree_s = ttk.Treeview(panel_izq, columns=cols_s, show="headings", height=10)
            tree_s.heading("k", text="k")
            tree_s.heading("p1", text=f"p1 ({probs['p1']:.2f})")
            tree_s.heading("p2", text=f"p2 ({probs['p2']:.2f})")
            tree_s.heading("p3", text=f"p3 ({probs['p3']:.2f})")
            for c_id, ancho in (("k", 45), ("p1", 105), ("p2", 105), ("p3", 105)):
                tree_s.column(c_id, width=ancho, anchor=tk.CENTER)

            series_p = datos["tablas_s10"][caso]
            for idx in range(10):
                tree_s.insert(
                    "",
                    tk.END,
                    values=(
                        idx + 1,
                        f"{series_p['p1'][idx]:.6f}",
                        f"{series_p['p2'][idx]:.6f}",
                        f"{series_p['p3'][idx]:.6f}",
                    ),
                )
            tree_s.pack(pady=(0, 10))

            caja_formula = tk.Frame(panel_izq, bg="#f8fafc", padx=10, pady=8, highlightbackground="#e2e8f0", highlightthickness=1)
            caja_formula.pack(fill=tk.X)

            tk.Label(
                caja_formula,
                text="Modelo Matemático:",
                font=("Segoe UI", 9, "bold"),
                bg="#f8fafc",
                fg="#334155",
            ).pack(anchor=tk.W)

            tk.Label(
                caja_formula,
                text=modelo.formula_texto,
                font=("Consolas", 9),
                bg="#f8fafc",
                fg="#0f172a",
                wraplength=330,
                justify=tk.LEFT,
            ).pack(anchor=tk.W, pady=(2, 0))

            panel_der = tk.Frame(tab_c, bg="#ffffff", padx=8, pady=8, highlightbackground="#cbd5e1", highlightthickness=1)
            panel_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

            fig_c = crear_figura_caso(caso, datos, k_override=k_val)
            figuras_activas.append(fig_c)
            canvas_c = FigureCanvasTkAgg(fig_c, master=panel_der)
            canvas_c.draw()

            toolbar = NavigationToolbar2Tk(canvas_c, panel_der)
            toolbar.update()
            canvas_c.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ======================================================================
        # PESTAÑA 5: COMPARADOR INTERACTIVO ENTRE ARQUITECTURAS (A vs B vs C)
        # ======================================================================
        tab_comp = ttk.Frame(notebook, padding=10)
        notebook.add(tab_comp, text="  5. Comparador A vs B vs C  ")

        barra_sel = tk.Frame(tab_comp, bg="#ffffff", padx=12, pady=8, highlightbackground="#cbd5e1", highlightthickness=1)
        barra_sel.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            barra_sel,
            text="Selecciona probabilidad para comparar los 3 protocolos en simultáneo:",
            font=("Segoe UI", 10, "bold"),
            bg="#ffffff",
            fg="#0f172a",
        ).pack(side=tk.LEFT, padx=(0, 12))

        prob_sel_var = tk.StringVar(master=root, value="p1")
        frame_plot_comp = tk.Frame(tab_comp, bg="#ffffff", padx=8, pady=8, highlightbackground="#cbd5e1", highlightthickness=1)
        frame_plot_comp.pack(fill=tk.BOTH, expand=True)

        def dibujar_comparativa() -> None:
            for w in frame_plot_comp.winfo_children():
                w.destroy()
            fig_comp = crear_figura_comparativa_prob(datos, prob_sel_var.get(), k_slider_var.get())
            figuras_activas.append(fig_comp)
            canvas_comp = FigureCanvasTkAgg(fig_comp, master=frame_plot_comp)
            canvas_comp.draw()
            toolbar_comp = NavigationToolbar2Tk(canvas_comp, frame_plot_comp)
            toolbar_comp.update()
            canvas_comp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        for nombre_p in ("p1", "p2", "p3"):
            rb = tk.Radiobutton(
                barra_sel,
                text=f"{nombre_p} = {probs[nombre_p]:.2f}",
                variable=prob_sel_var,
                value=nombre_p,
                command=dibujar_comparativa,
                font=("Consolas", 10, "bold"),
                bg="#ffffff",
                fg="#0f172a",
                activebackground="#ffffff",
            )
            rb.pack(side=tk.LEFT, padx=8)

        dibujar_comparativa()

        if 0 <= pestana_activa < len(notebook.tabs()):
            notebook.select(pestana_activa)

    def ejecutar_recalculo(_event=None) -> None:
        try:
            c_max_nuevo = float(cmax_var.get().replace(",", "."))
            if c_max_nuevo <= 0:
                raise ValueError("El umbral C_max debe ser mayor que 0.")
            nuevos_datos = ejecutar_analisis(rut_var.get(), c_max=c_max_nuevo)
            datos_estado["actual"] = nuevos_datos
            figs_png = guardar_graficos_png(nuevos_datos)
            for f in figs_png.values():
                plt.close(f)
            print("\n" + "=" * 82)
            print("RECÁLCULO EN VIVO DESDE LA APLICACIÓN")
            imprimir_reporte(nuevos_datos)
            renderizar_todo(nuevos_datos)
            estado_var.set(f"Actualizado • RUT {nuevos_datos['rut_8_digitos']} • PNGs guardados")
        except ValueError as err:
            messagebox.showerror("Dato inválido", str(err))

    btn_calc = tk.Button(
        panel_ctrl,
        text="Recalcular y Proyectar",
        command=ejecutar_recalculo,
        font=("Segoe UI", 10, "bold"),
        bg="#2563eb",
        fg="#ffffff",
        activebackground="#1d4ed8",
        activeforeground="#ffffff",
        relief="flat",
        padx=14,
        pady=4,
        cursor="hand2",
    )
    btn_calc.pack(side=tk.LEFT)

    tk.Label(
        panel_ctrl,
        textvariable=estado_var,
        font=("Segoe UI", 9, "bold"),
        bg="#ffffff",
        fg="#16a34a",
    ).pack(side=tk.RIGHT)

    entry_rut.bind("<Return>", ejecutar_recalculo)
    entry_cmax.bind("<Return>", ejecutar_recalculo)

    renderizar_todo(datos_iniciales)

    def al_cerrar() -> None:
        limpiar_figuras()
        plt.close("all")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", al_cerrar)
    root.mainloop()
    return True


def main() -> None:
    datos_ejecucion = ejecutar_analisis(RUT_POR_DEFECTO, c_max=C_MAX)
    imprimir_reporte(datos_ejecucion)
    figuras = guardar_graficos_png(datos_ejecucion)

    if USAR_CLI_FORZADO:
        for f in figuras.values():
            plt.close(f)
        return

    gui_iniciada = lanzar_interfaz_grafica(datos_ejecucion, figuras)
    if not gui_iniciada:
        if matplotlib.get_backend().lower() != "agg":
            plt.show()
        else:
            for f in figuras.values():
                plt.close(f)


if __name__ == "__main__":
    main()