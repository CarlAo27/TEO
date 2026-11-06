import numpy as np
import pandas as pd
import streamlit as st

def composicion_max_min(A, B):
    """Calcula la composición máx-mín de dos matrices."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A y B deben ser matrices.")

    if A.shape[1] != B.shape[0]:
        raise ValueError(
            "Las dimensiones no son compatibles para la composición."
        )

    resultado = np.zeros((A.shape[0], B.shape[1]))

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            resultado[i, j] = np.max(
                np.minimum(A[i, :], B[:, j])
            )

    return resultado

def calcular_hasta_estabilizacion(M, max_iter=100, etiquetas=None, tol=1e-6):
    """
    Calcula potencias máx-mín de M hasta que la matriz se estabilice
    o se alcance max_iter.
    """
    M = np.asarray(M, dtype=float)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M debe ser una matriz cuadrada.")

    n = M.shape[0]

    if etiquetas is None:
        etiquetas = [f"a{i + 1}" for i in range(n)]

    if len(etiquetas) != n:
        raise ValueError(
            f"Se necesitan {n} etiquetas para una matriz de {n} × {n}."
        )

    historial = [M.copy()]
    M_actual = M.copy()
    estabilizado = False
    k_estabilizacion = None

    print("=== MATRIZ INICIAL M ===")
    print(pd.DataFrame(M, index=etiquetas, columns=etiquetas))

    for k in range(2, max_iter + 1):
        M_siguiente = composicion_max_min(M_actual, M)
        historial.append(M_siguiente.copy())

        print(f"\n=== MATRIZ M^{k} ===")
        print(pd.DataFrame(M_siguiente, index=etiquetas, columns=etiquetas))

        if np.allclose(M_siguiente, M_actual, atol=tol):
            estabilizado = True
            k_estabilizacion = k
            M_actual = M_siguiente
            break

        M_actual = M_siguiente

    if estabilizado:
        print(f"\nLa matriz se estabilizó en k = {k_estabilizacion}.")
    else:
        print(f"\nNo se estabilizó después de {max_iter} generaciones.")

    M_estrella = historial[-1]
    matriz_efectos = np.round(M_estrella - M, 6)

    print("\n=== EFECTOS OLVIDADOS TOTALES (M* - M) ===")
    print(pd.DataFrame(
        matriz_efectos,
        index=etiquetas,
        columns=etiquetas
    ))

    return {
        "estabilizado": estabilizado,
        "k_estabilizacion": k_estabilizacion,
        "historial": historial,
        "M_estrella": M_estrella,
        "matriz_efectos": matriz_efectos
    }


def explicar_efectos_olvidados(M, M_estrella, etiquetas, tolerancia=1e-6):
    """Busca caminos indirectos que expliquen los incrementos de M* - M."""
    M = np.asarray(M, dtype=float)
    M_estrella = np.asarray(M_estrella, dtype=float)

    if M.shape != M_estrella.shape:
        raise ValueError("M y M_estrella deben tener las mismas dimensiones.")

    n = M.shape[0]

    if M.shape != (n, n) or len(etiquetas) != n:
        raise ValueError("La matriz debe ser cuadrada y tener una etiqueta por variable.")

    efectos = np.round(M_estrella - M, 6)
    resultados = []

    def buscar_caminos(origen, destino):
        caminos = []

        def dfs(actual, visitados, valores):
            if actual == destino and len(visitados) >= 3:
                caminos.append((min(valores), visitados.copy()))
                return

            for siguiente in range(n):
                if siguiente in visitados:
                    continue

                valor = M[actual, siguiente]
                if valor > 0:
                    dfs(
                        siguiente,
                        visitados + [siguiente],
                        valores + [valor]
                    )

        dfs(origen, [origen], [])
        return caminos

    for i in range(n):
        for j in range(n):
            incremento = efectos[i, j]

            if i == j or incremento <= tolerancia:
                continue

            caminos = buscar_caminos(i, j)
            caminos.sort(key=lambda camino: camino[0], reverse=True)

            if caminos:
                fuerza, indices = caminos[0]
                camino_texto = " → ".join(etiquetas[k] for k in indices)
            else:
                fuerza = 0
                camino_texto = "No se encontró un camino indirecto."

            interpretacion = (
                f"La relación de {etiquetas[i]} a {etiquetas[j]} "
                f"aumentó {incremento:.2f}, de {M[i, j]:.2f} "
                f"a {M_estrella[i, j]:.2f}."
            )

            if caminos:
                interpretacion += (
                    f" El camino indirecto más intenso encontrado es "
                    f"{camino_texto}, con intensidad MAX-MIN de {fuerza:.2f}."
                )

            resultados.append({
                "Variable de origen": etiquetas[i],
                "Variable de destino": etiquetas[j],
                "Valor inicial": M[i, j],
                "Valor estabilizado": M_estrella[i, j],
                "Incremento": incremento,
                "Camino indirecto": camino_texto,
                "Intensidad del camino": fuerza,
                "Interpretación": interpretacion
            })

    tabla = pd.DataFrame(resultados)

    if tabla.empty:
        print("No se encontraron incrementos positivos.")
    else:
        tabla = tabla.sort_values(
            "Incremento", ascending=False
        ).reset_index(drop=True)

        print("=== INTERPRETACIÓN DE EFECTOS OLVIDADOS ===")
        for _, fila in tabla.iterrows():
            print(
                f"\n{fila['Variable de origen']} → "
                f"{fila['Variable de destino']}"
            )
            print(f"Incremento: {fila['Incremento']:.2f}")
            print(f"Camino: {fila['Camino indirecto']}")
            print(fila["Interpretación"])

    return tabla


M_ejemplo = [
    [1.0, 0.7, 0.4, 0.2, 0.1, 0.1],
    [0.5, 1.0, 0.7, 0.3, 0.2, 0.2],
    [0.2, 0.4, 1.0, 0.7, 0.4, 0.3],
    [0.1, 0.2, 0.5, 1.0, 0.8, 0.4],
    [0.1, 0.2, 0.3, 0.5, 1.0, 0.7],
    [0.2, 0.2, 0.3, 0.4, 0.6, 1.0]
]

etiquetas = [
    "Capacitación",
    "Motivación",
    "Productividad",
    "Calidad",
    "Satisfacción",
    "Rentabilidad"
]

resultado = calcular_hasta_estabilizacion(
    M_ejemplo,
    max_iter=50,
    etiquetas=etiquetas
)

M_estrella = resultado["M_estrella"]

tabla_interpretaciones = explicar_efectos_olvidados(
    M=M_ejemplo,
    M_estrella=M_estrella,
    etiquetas=etiquetas
)

st.title("Análisis de efectos olvidados")
st.dataframe(tabla_interpretaciones, use_container_width=True)