from dataclasses import dataclass
from typing import Callable
import matplotlib.pyplot as plt


# ==============================================================================
# 1. CONFIGURACIÓN DEL EQUIPO Y PARÁMETROS
# ==============================================================================
RUT_POR_DEFECTO = "24.365.895-K"  # Reemplazar por el RUT del integrante elegido
INTEGRANTES = ["Mauricio Valdés", "Compañero 2", "Compañero 3"]
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

# --- RAMA: feature/caso-a (TAREA DE TU COMPAÑERO) ---
def termino_caso_a(i: int, p: float) -> float:
    """
    Caso A (WiFi estándar): i * (1 - p)^(i - 1) * p
    TODO (Compañero): Reemplazar el raise por el retorno de la fórmula.
    """
    raise NotImplementedError("Pendiente: implementar Caso A en rama feature/caso-a")


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
    return {"p1": p1, "p2": p2, "p3": p3}


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


def ejecutar_analisis(rut_raw: str) -> dict:
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
                    cumple_umbral=conv and (lim_emp <= C_MAX),
                )
            )

    return {
        "rut_ingresado": rut_raw,
        "rut_8_digitos": rut_limpio,
        "digitos": digitos,
        "probabilidades": probs,
        "casos_disponibles": casos_disponibles,
        "tablas_s10": tablas_s10,
        "resultados": resultados,
    }


# ==============================================================================
# 4. IMPRESIÓN EN CONSOLA Y GRÁFICOS AUTOMÁTICOS
# ==============================================================================
def imprimir_reporte(datos: dict) -> None:
    probs = datos["probabilidades"]
    col_p1 = f"p1={probs['p1']:.2f}"
    col_p2 = f"p2={probs['p2']:.2f}"
    col_p3 = f"p3={probs['p3']:.2f}"

    print("=" * 80)
    print("EID N°2 - MATE1133: ANÁLISIS DE CONVERGENCIA PROTOCOLO ARQ")
    print("=" * 80)
    print(f"Integrantes : {', '.join(INTEGRANTES)}")
    print(f"RUT elegido : {datos['rut_ingresado']} -> Dígitos = {datos['digitos']}")
    print(f"Parámetros  : p1 = {probs['p1']:.4f} | p2 = {probs['p2']:.4f} | p3 = {probs['p3']:.4f}")
    print("=" * 80)

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

    print("\n" + "=" * 80)
    print(f"CONVERGENCIA Y TABLA DE VIABILIDAD (UMBRAL C_max = {C_MAX:.1f})")
    print("=" * 80)
    print(f"{'Caso':<6} | {'Probabilidad':<13} | {'¿Converge?':<11} | {'Límite Empírico':>16} | {'Viabilidad':<12}")
    print("-" * 80)
    for r in datos["resultados"]:
        estado_conv = "Sí" if r.converge else "No"
        estado_viab = "CUMPLE" if r.cumple_umbral else "NO CUMPLE"
        print(
            f"{r.caso:<6} | {r.nombre_p} ({r.p:.2f})     | {estado_conv:<11} | "
            f"{r.limite_empirico:>16.6f} | {estado_viab:<12}"
        )

    viables = [r for r in datos["resultados"] if r.cumple_umbral]
    print("-" * 80)
    if viables:
        mejor = min(viables, key=lambda r: (r.limite_empirico, r.iteraciones_conv))
        print(
            f"RECOMENDACIÓN: Caso {mejor.caso} con {mejor.nombre_p} = {mejor.p:.2f} "
            f"(límite empírico = {mejor.limite_empirico:.6f} <= {C_MAX:.1f}, "
            f"se estabiliza en k = {mejor.iteraciones_conv} iteraciones)."
        )
    print("=" * 80)


def generar_graficos(datos: dict) -> None:
    colores = {"p1": "#1f77b4", "p2": "#ff7f0e", "p3": "#2ca02c"}
    for caso in datos["casos_disponibles"]:
        modelo = MODELOS_ARQ[caso]
        res_caso = [r for r in datos["resultados"] if r.caso == caso]
        k_max_plot = min(max(max(r.iteraciones_conv for r in res_caso) // 2, 25), 90)
        eje_k = list(range(1, k_max_plot + 1))

        plt.figure(figsize=(8.5, 4.6))
        for r in res_caso:
            color = colores[r.nombre_p]
            y_vals = calcular_sumas_parciales(caso, r.p, k_max_plot)
            plt.plot(
                eje_k,
                y_vals,
                marker="o",
                markersize=3,
                linewidth=1.8,
                color=color,
                label=f"{r.nombre_p} = {r.p:.2f} (Límite ≈ {r.limite_empirico:.4f})",
            )
            plt.axhline(y=r.limite_empirico, color=color, linestyle="--", alpha=0.65)

        plt.title(f"Caso {caso}: {modelo.nombre}\n{modelo.formula_texto}", fontsize=10.5)
        plt.xlabel("Número de términos acumulados (k)")
        plt.ylabel("Costo / Transmisiones esperadas (S_k)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="best", fontsize=8.5)
        plt.tight_layout()
        plt.savefig(f"grafico_caso_{caso}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    datos_ejecucion = ejecutar_analisis(RUT_POR_DEFECTO)
    imprimir_reporte(datos_ejecucion)
    generar_graficos(datos_ejecucion)