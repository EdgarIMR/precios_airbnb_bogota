"""Prepara los insumos del tablero BogoPrice.

1. Entrena el modelo servible (Ridge, alpha=0.1, variables v1 = config.FEATURES),
   el mejor modelo lineal registrado en MLflow (experimento bogoprice-lineales).
2. Entrena dos modelos de cuantil (P10 y P90) para el rango sugerido del 80 %.
3. Guarda todo en models/modelo.joblib con bogoprice.exportar.guardar_modelo,
   el mismo formato que usará la API en la semana 6. Se agregan las medias de
   las variables transformadas en entrenamiento, que el tablero usa para la
   contribución de cada variable en COP.
4. Exporta los insumos de la vista de monitoreo:
   - tablero/datos/anuncios_tablero.csv.gz: anuncios de septiembre 2026 con su
     precio estimado y rango (comparables y gráfico estimado vs. publicado).
   - tablero/datos/metricas_modelo.json: MdAPE por captura y por localidad,
     cobertura del rango y datos de la captura evaluada.

Uso (desde la raíz del repo):
    pip install -e . && pip install -r tablero/requirements.txt
    python tablero/preparar_tablero.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
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
from bogoprice.metrics import cobertura, mdape, metricas_precio

VERSION = "v1.0 (Ridge α=0,1)"
DIR_DATOS = Path(__file__).resolve().parent / "datos"
COLUMNAS_TABLERO = ["id", config.TARGET] + config.FEATURES


def main() -> None:
    train, test, limite = cargar_train_test()
    fecha_captura = str(test["last_scraped"].max())
    train, test = construir_features(train), construir_features(test)
    localidades = localidades_habilitadas(train)
    train = filtrar_localidades(train, localidades)
    test = filtrar_localidades(test, localidades)

    X, y = train[config.FEATURES], train[config.TARGET_LOG]

    # Modelo puntual: el mejor lineal de MLflow (Grupo6_ridge_a0.1)
    modelo = construir_pipeline(Ridge(alpha=0.1)).fit(X, y)

    # Rango sugerido del 80 %: cuantiles 0.1 y 0.9 del log-precio
    cuantiles = {}
    for q in (0.1, 0.9):
        gbm = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, random_state=0)
        cuantiles[q] = construir_pipeline(gbm, escalar=False).fit(X, y)

    # Evaluación fuera de tiempo: entrenado con junio, evaluado con septiembre
    Xt = test[config.FEATURES]
    test = test.assign(
        precio_estimado=np.exp(modelo.predict(Xt)),
        rango_inf=np.exp(cuantiles[0.1].predict(Xt)),
        rango_sup=np.exp(cuantiles[0.9].predict(Xt)),
    )
    m = metricas_precio(test[config.TARGET], test["precio_estimado"])
    m["cobertura_rango"] = cobertura(test[config.TARGET], test["rango_inf"], test["rango_sup"])
    m_train = metricas_precio(train[config.TARGET], np.exp(modelo.predict(X)))
    m_train["cobertura_rango"] = cobertura(
        train[config.TARGET], np.exp(cuantiles[0.1].predict(X)), np.exp(cuantiles[0.9].predict(X))
    )
    por_localidad = {
        loc: round(mdape(g[config.TARGET], g["precio_estimado"]), 2)
        for loc, g in test.groupby("neighbourhood_cleansed")
    }
    print(f"P99 junio: ${limite:,.0f} | localidades: {localidades}")
    print("septiembre:", {k: round(v, 3) for k, v in m.items()})
    print("junio (entrenamiento):", {k: round(v, 3) for k, v in m_train.items()})

    ruta = guardar_modelo(modelo, VERSION, cuantiles[0.1], cuantiles[0.9])
    paquete = joblib.load(ruta)
    paquete["medias_transformadas"] = np.asarray(
        modelo.named_steps["prep"].transform(X).mean(axis=0)
    ).ravel()
    joblib.dump(paquete, ruta)
    print(f"Modelo guardado en {ruta}")

    DIR_DATOS.mkdir(parents=True, exist_ok=True)
    cols = COLUMNAS_TABLERO + ["precio_estimado", "rango_inf", "rango_sup"]
    test[cols].round(0).to_csv(DIR_DATOS / "anuncios_tablero.csv.gz", index=False, compression="gzip")
    metricas = {
        "version": VERSION,
        "captura_entrenamiento": "Junio 2026",
        "captura_evaluacion": "Septiembre 2026",
        "fecha_captura_evaluacion": fecha_captura,
        "anuncios_evaluacion": int(len(test)),
        "meta_mdape": 20.0,
        "umbral_reentrenamiento": 25.0,
        "evaluacion": {k: round(float(v), 3) for k, v in m.items()},
        "entrenamiento": {k: round(float(v), 3) for k, v in m_train.items()},
        "mdape_por_localidad": por_localidad,
    }
    (DIR_DATOS / "metricas_modelo.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False))
    print(f"Insumos del tablero en {DIR_DATOS} ({len(test)} anuncios)")


if __name__ == "__main__":
    main()
