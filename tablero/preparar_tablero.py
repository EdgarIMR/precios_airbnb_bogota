"""Prepara los insumos del tablero BogoPrice a partir del modelo final de Edgar (LightGBM).

1. Si no existen api/modelos/modelo_v2.joblib y modelo_v2_geo.joblib, los entrena con
   bogoprice.modelo_final (lo mismo que hace el cuaderno 09, sin registrar en MLflow).
2. Repite la validación fuera de tiempo (entrena con junio, evalúa con septiembre) para el
   panel de monitoreo. Las métricas se miden en los anuncios nuevos de septiembre, los que
   no estaban en junio: es la prueba honesta, porque un modelo de árboles puede "recordar"
   un anuncio que ya vio. Se incluye el Ridge v1 para comparar versiones.
3. Exporta:
   - tablero/datos/anuncios_tablero.csv.gz: anuncios de septiembre (comparables de la
     calculadora y gráfico estimado frente a publicado).
   - tablero/datos/metricas_modelo.json: métricas del panel de monitoreo.

Uso (desde la raíz del repo):
    pip install -e . && pip install -r tablero/requirements.txt
    python tablero/preparar_tablero.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from bogoprice import config, modelo_final, variables_v2
from bogoprice.experimentos import preparar_datos
from bogoprice.features import construir_pipeline
from bogoprice.metrics import cobertura, mdape

DIR_DATOS = Path(__file__).resolve().parent / "datos"


def asegurar_modelos(datos, reentrenar: bool = False) -> None:
    """Entrena los paquetes finales si todavía no están en api/modelos (o si reentrenar=True)."""
    faltan = [g for g in (False, True) if reentrenar or not modelo_final.ruta_paquete(g).exists()]
    if not faltan:
        return
    print("Entrenando el modelo final (junio + septiembre)...")
    df, info = modelo_final.preparar_entrenamiento_final()
    for geo in faltan:
        validacion = modelo_final.validar(geo, datos)
        paquete = modelo_final.entrenar_paquete(geo, df, info, validacion)
        print("  guardado", modelo_final.guardar_paquete(paquete))
    modelo_final.escribir_requirements_api(paquete)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Insumos del tablero BogoPrice")
    parser.add_argument("--reentrenar", action="store_true",
                        help="Vuelve a entrenar api/modelos aunque ya existan")
    args = parser.parse_args()

    datos = preparar_datos()
    asegurar_modelos(datos, reentrenar=args.reentrenar)
    test, nuevo = datos.test.copy(), datos.es_nuevo
    y = datos.y_test_cop

    # Validación de las dos versiones de LightGBM (modelos entrenados solo con junio)
    evaluacion, predicciones = {}, {}
    for geo in (False, True):
        version = modelo_final.version_de(geo)
        precio, q_inf, q_sup = modelo_final._entrenar(datos.train, version)
        X = test[variables_v2.columnas(version)]
        pred = np.exp(precio.predict(X))
        inf = np.minimum(np.exp(q_inf.predict(X)), pred)
        sup = np.maximum(np.exp(q_sup.predict(X)), pred)
        evaluacion[version] = {
            "mdape_nuevos": mdape(y[nuevo], pred[nuevo]),
            "cobertura_nuevos": cobertura(y[nuevo], inf[nuevo], sup[nuevo]),
            "mae_nuevos": float(np.mean(np.abs(y[nuevo] - pred[nuevo]))),
            "mdape_todos": mdape(y, pred),
        }
        predicciones[version] = pred   # para el error por localidad y la dispersión

    # Modelos de referencia en el mismo esquema (entrenados con junio, medidos en anuncios nuevos):
    # la mediana por localidad y tipo de alojamiento (línea base) y el mejor modelo lineal
    # del cuaderno 02 (Ridge con alpha=0,1 y las 14 variables del formulario)
    claves = ["neighbourhood_cleansed", "room_type"]
    medianas = datos.train.groupby(claves)[config.TARGET].median().rename("pred")
    pred_mediana = (test[claves].join(medianas, on=claves)["pred"]
                    .fillna(datos.train[config.TARGET].median()).values)
    ridge = construir_pipeline(Ridge(alpha=0.1)).fit(datos.train[config.FEATURES], datos.y_train)
    pred_ridge = np.exp(ridge.predict(test[config.FEATURES]))

    # Precio estimado por cada versión (el tablero deja elegir cuál graficar)
    for version, pred in predicciones.items():
        test[f"precio_{version}"] = np.round(pred, -2)
    test["es_nuevo"] = nuevo.astype(int)
    nuevos = test[test["es_nuevo"] == 1]
    por_localidad = {
        version: {loc: {"mdape": round(mdape(g[config.TARGET], g[f"precio_{version}"]), 2), "n": int(len(g))}
                  for loc, g in nuevos.groupby("neighbourhood_cleansed")}
        for version in predicciones
    }

    final = joblib.load(modelo_final.ruta_paquete(False))
    metricas = {
        "fecha_captura_evaluacion": str(datos.test["last_scraped"].max()),
        "anuncios_evaluacion": int(len(test)),
        "anuncios_nuevos": int(nuevo.sum()),
        "meta_mdape": 20.0,
        "umbral_reentrenamiento": 25.0,
        "evaluacion": {k: {m: round(float(v), 3) for m, v in d.items()} for k, d in evaluacion.items()},
        "versiones": [
            {"nombre": "Mediana por localidad y tipo",
             "mdape_nuevos": round(mdape(y[nuevo], pred_mediana[nuevo]), 2)},
            {"nombre": "Ridge", "mdape_nuevos": round(mdape(y[nuevo], pred_ridge[nuevo]), 2)},
            {"nombre": "LightGBM sin ubicación",
             "mdape_nuevos": round(evaluacion["v2"]["mdape_nuevos"], 2)},
            {"nombre": "LightGBM con ubicación",
             "mdape_nuevos": round(evaluacion["v2_geo"]["mdape_nuevos"], 2)},
        ],
        "mdape_por_localidad": por_localidad,
        "entrenamiento_final": final["entrenamiento"],
        "percentiles_rango": list(final["percentiles"]),
        # El tablero lee estos valores de aquí para no depender del paquete bogoprice
        "min_anuncios_localidad": config.MIN_ANUNCIOS_LOCALIDAD,
        "parametros_modelo": final["parametros_precio"],
    }
    print(json.dumps(metricas["versiones"], ensure_ascii=False))
    print(json.dumps(metricas["evaluacion"], ensure_ascii=False))

    DIR_DATOS.mkdir(parents=True, exist_ok=True)
    cols = ["id", config.TARGET, "neighbourhood_cleansed", "room_type", "accommodates",
            "latitude", "longitude", "precio_v2", "precio_v2_geo", "es_nuevo"]
    test[cols].to_csv(DIR_DATOS / "anuncios_tablero.csv.gz", index=False, compression="gzip")
    # UTF-8 explícito: en Windows el valor por defecto es cp1252 y el contenedor (Linux) no lo lee
    (DIR_DATOS / "metricas_modelo.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print(f"Insumos del tablero en {DIR_DATOS} ({len(test)} anuncios, {int(nuevo.sum())} nuevos)")


if __name__ == "__main__":
    main()
