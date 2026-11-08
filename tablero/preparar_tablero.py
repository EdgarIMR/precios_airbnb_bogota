"""Prepara los insumos del tablero BogoPrice.

1. Entrena el modelo servible (Ridge, alpha=0.1, variables v1 = config.FEATURES),
   el mejor modelo lineal registrado en MLflow (experimento bogoprice-lineales).
2. Entrena dos modelos de cuantil (P10 y P90) para el rango competitivo.
3. Guarda todo en models/modelo.joblib con bogoprice.exportar.guardar_modelo,
   el mismo formato que usará la API en la semana 6.
4. Exporta una tabla liviana de anuncios de septiembre 2026 para la parte
   descriptiva del tablero (tablero/datos/anuncios_tablero.csv.gz).

Uso (desde la raíz del repo):
    pip install -e . && pip install -r tablero/requirements.txt
    python tablero/preparar_tablero.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from bogoprice import config
from bogoprice.data_loader import cargar_train_test
from bogoprice.exportar import guardar_modelo
from bogoprice.features import (
    construir_features,
    construir_pipeline,
    filtrar_localidades,
    localidades_habilitadas,
)
from bogoprice.metrics import cobertura, metricas_precio

SALIDA_DATOS = Path(__file__).resolve().parent / "datos" / "anuncios_tablero.csv.gz"
COLUMNAS_TABLERO = (
    ["id", "latitude", "longitude", config.TARGET, "property_type"] + config.FEATURES
)


def main() -> None:
    train, test, limite = cargar_train_test()
    train, test = construir_features(train), construir_features(test)
    localidades = localidades_habilitadas(train)
    train = filtrar_localidades(train, localidades)
    test = filtrar_localidades(test, localidades)

    X, y = train[config.FEATURES], train[config.TARGET_LOG]

    # Modelo puntual: el mejor lineal de MLflow (Grupo6_ridge_a0.1)
    modelo = construir_pipeline(Ridge(alpha=0.1)).fit(X, y)

    # Rango competitivo: cuantiles 0.1 y 0.9 del log-precio
    cuantiles = {}
    for q in (0.1, 0.9):
        gbm = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=300, random_state=0
        )
        cuantiles[q] = construir_pipeline(gbm, escalar=False).fit(X, y)

    # Evaluación en septiembre (anuncios fuera del entrenamiento temporal)
    Xt = test[config.FEATURES]
    pred = np.exp(modelo.predict(Xt))
    inf, sup = np.exp(cuantiles[0.1].predict(Xt)), np.exp(cuantiles[0.9].predict(Xt))
    m = metricas_precio(test[config.TARGET], pred)
    m["cobertura_rango"] = cobertura(test[config.TARGET], inf, sup)
    print(f"P99 junio: ${limite:,.0f} | localidades: {localidades}")
    print({k: round(v, 3) for k, v in m.items()})

    ruta = guardar_modelo(modelo, "ridge_a0.1_v1", cuantiles[0.1], cuantiles[0.9])
    print(f"Modelo guardado en {ruta}")

    SALIDA_DATOS.parent.mkdir(parents=True, exist_ok=True)
    test[COLUMNAS_TABLERO].to_csv(SALIDA_DATOS, index=False, compression="gzip")
    print(f"Datos del tablero: {SALIDA_DATOS} ({len(test)} anuncios)")


if __name__ == "__main__":
    main()
