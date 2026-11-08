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


def asegurar_modelos(datos) -> None:
    """Entrena los paquetes finales si todavía no están en api/modelos."""
    faltan = [g for g in (False, True) if not modelo_final.ruta_paquete(g).exists()]
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
    datos = preparar_datos()
    asegurar_modelos(datos)
    test, nuevo = datos.test.copy(), datos.es_nuevo
    y = datos.y_test_cop

    # Validación de las dos versiones de LightGBM (modelos entrenados solo con junio)
    evaluacion, pred_v2 = {}, None
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
        if not geo:
            pred_v2 = pred

    # Ridge v1 (la línea base lineal del grupo) en el mismo esquema
    ridge = construir_pipeline(Ridge(alpha=0.1)).fit(datos.train[config.FEATURES], datos.y_train)
    pred_ridge = np.exp(ridge.predict(test[config.FEATURES]))

    test["precio_validacion"] = np.round(pred_v2, -2)
    test["es_nuevo"] = nuevo.astype(int)
    por_localidad = {
        loc: {"mdape": round(mdape(g[config.TARGET], g["precio_validacion"]), 2), "n": int(len(g))}
        for loc, g in test[test["es_nuevo"] == 1].groupby("neighbourhood_cleansed")
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
            {"nombre": "Ridge v1", "mdape_nuevos": round(mdape(y[nuevo], pred_ridge[nuevo]), 2)},
            {"nombre": "LightGBM v2", "mdape_nuevos": round(evaluacion["v2"]["mdape_nuevos"], 2)},
            {"nombre": "LightGBM v2 + ubicación", "mdape_nuevos": round(evaluacion["v2_geo"]["mdape_nuevos"], 2)},
        ],
        "mdape_por_localidad": por_localidad,
        "entrenamiento_final": final["entrenamiento"],
        "percentiles_rango": list(final["percentiles"]),
    }
    print(json.dumps(metricas["versiones"], ensure_ascii=False))
    print(json.dumps(metricas["evaluacion"], ensure_ascii=False))

    DIR_DATOS.mkdir(parents=True, exist_ok=True)
    cols = ["id", config.TARGET, "neighbourhood_cleansed", "room_type", "accommodates",
            "latitude", "longitude", "precio_validacion", "es_nuevo"]
    test[cols].to_csv(DIR_DATOS / "anuncios_tablero.csv.gz", index=False, compression="gzip")
    (DIR_DATOS / "metricas_modelo.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False))
    print(f"Insumos del tablero en {DIR_DATOS} ({len(test)} anuncios, {int(nuevo.sum())} nuevos)")


if __name__ == "__main__":
    main()
