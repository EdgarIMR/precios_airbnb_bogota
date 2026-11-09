"""Entrenamiento del modelo final de BogoPrice (LightGBM) para la API y el tablero.

Define la configuración elegida en el cuaderno 04 y entrena, para cada conjunto
de variables, tres modelos:

    modelo  precio de referencia (pérdida l1: estima la mediana del precio)
    q_inf   límite inferior del rango sugerido (percentil 7,5)
    q_sup   límite superior del rango sugerido (percentil 92,5)

Hay dos versiones, según el booleano ``usar_coordenadas``:

    False -> conjunto v2      (el anfitrión no indica su ubicación exacta)
    True  -> conjunto v2_geo  (v2 + latitud y longitud, p. ej. desde un mapa)

Los paquetes se guardan en api/modelos/. La carpeta api/ (con api/predictor.py)
funciona sola: no necesita este paquete para predecir. El paquete conserva las claves
de ``exportar.guardar_modelo`` (modelo, q_inf, q_sup, version) y agrega lo que la API
y el tablero necesitan.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from . import config, variables_v2
from .data_loader import cargar_y_limpiar, filtrar_precio, limite_atipicos
from .features import construir_features, filtrar_localidades, localidades_habilitadas
from .metrics import cobertura, mdape

# --- Configuración elegida en el cuaderno 04 (grilla objective x learning_rate) -----------
BASE_LGBM = {"n_estimators": 1500, "num_leaves": 63, "max_depth": -1, "min_child_samples": 20,
             "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7,
             "reg_lambda": 0.0, "reg_alpha": 0.0}

PARAMETROS_PRECIO = {
    "v2": BASE_LGBM | {"objective": "l1", "learning_rate": 0.02},      # 19,33 % MdAPE nuevos
    "v2_geo": BASE_LGBM | {"objective": "l1", "learning_rate": 0.01},  # 17,40 % MdAPE nuevos
}

# Rango sugerido: regresión cuantílica (cuaderno 07)
PERCENTILES = (0.075, 0.925)
PARAMETROS_RANGO = {"n_estimators": 600, "learning_rate": 0.03, "num_leaves": 31,
                    "min_child_samples": 30, "subsample": 0.8, "subsample_freq": 1,
                    "colsample_bytree": 0.7}

CAPTURAS_FINALES = ("2026-06", "2026-09")


def version_de(usar_coordenadas: bool) -> str:
    """Conjunto de variables según si se usan o no las coordenadas."""
    return "v2_geo" if usar_coordenadas else "v2"


# Los modelos se guardan dentro de la carpeta api/, que funciona sola para el tablero
CARPETA_API = config.ROOT / "api"


def ruta_paquete(usar_coordenadas: bool) -> Path:
    """Ruta estándar del paquete: api/modelos/modelo_v2.joblib o modelo_v2_geo.joblib."""
    return CARPETA_API / "modelos" / f"modelo_{version_de(usar_coordenadas)}.joblib"


def escribir_requirements_api(paquete: dict) -> Path:
    """Escribe api/requirements.txt con las versiones exactas usadas en el entrenamiento."""
    v = paquete["versiones"]
    ruta = CARPETA_API / "requirements.txt"
    ruta.write_text(
        "# Versiones con las que se entrenaron los modelos (generado por el cuaderno 09)\n"
        f"lightgbm=={v['lightgbm']}\nscikit-learn=={v['scikit-learn']}\n"
        f"pandas=={v['pandas']}\nnumpy=={v['numpy']}\njoblib\n")
    return ruta


def _pipelines(version: str):
    """Tres pipelines sin entrenar: precio, límite inferior y límite superior."""
    precio = variables_v2.construir_pipeline(
        lgb.LGBMRegressor(verbose=-1, random_state=0, **PARAMETROS_PRECIO[version]), version)
    cuantiles = [
        variables_v2.construir_pipeline(
            lgb.LGBMRegressor(objective="quantile", alpha=a, verbose=-1, random_state=0,
                              **PARAMETROS_RANGO), version)
        for a in PERCENTILES
    ]
    return precio, cuantiles[0], cuantiles[1]


def _entrenar(df: pd.DataFrame, version: str):
    cols = variables_v2.columnas(version)
    X, y = df[cols], df[config.TARGET_LOG]
    precio, q_inf, q_sup = _pipelines(version)
    return precio.fit(X, y), q_inf.fit(X, y), q_sup.fit(X, y)


def _evaluar(modelos, df: pd.DataFrame, version: str, es_nuevo: np.ndarray) -> dict:
    precio, q_inf, q_sup = modelos
    cols = variables_v2.columnas(version)
    y = df[config.TARGET].values
    pred = np.exp(precio.predict(df[cols]))
    inf, sup = np.exp(q_inf.predict(df[cols])), np.exp(q_sup.predict(df[cols]))
    inf, sup = np.minimum(inf, pred), np.maximum(sup, pred)
    return {
        "mdape": mdape(y, pred),
        "mdape_nuevos": mdape(y[es_nuevo], pred[es_nuevo]),
        "cobertura": cobertura(y, inf, sup),
        "cobertura_nuevos": cobertura(y[es_nuevo], inf[es_nuevo], sup[es_nuevo]),
        "ancho_mediano_cop": float(np.median(sup - inf)),
    }


def validar(usar_coordenadas: bool, datos=None) -> dict:
    """Entrena con junio y evalúa con septiembre (mismo esquema de los cuadernos 03 a 07).

    Devuelve las métricas de validación del precio y del rango."""
    from .experimentos import preparar_datos  # import local: solo se usa aquí

    datos = datos or preparar_datos()
    version = version_de(usar_coordenadas)
    modelos = _entrenar(datos.train, version)
    return _evaluar(modelos, datos.test, version, datos.es_nuevo)


def preparar_entrenamiento_final(capturas=CAPTURAS_FINALES) -> tuple[pd.DataFrame, dict]:
    """Une las capturas para el entrenamiento final.

    Si un anuncio aparece en varias capturas, se conserva su observación más reciente,
    para no contarlo dos veces. El límite P99, las localidades habilitadas y las
    tipologías frecuentes se recalculan con los datos unidos."""
    partes = [cargar_y_limpiar(c).assign(captura=c) for c in capturas]
    df = (pd.concat(partes, ignore_index=True)
          .sort_values("captura")
          .drop_duplicates("id", keep="last"))
    limite = limite_atipicos(df)
    df = construir_features(filtrar_precio(df, limite))
    localidades = localidades_habilitadas(df)
    df = filtrar_localidades(df, localidades)
    tipologias = variables_v2.tipologias_frecuentes(df)
    df = variables_v2.agregar_variables_v2(df, tipologias)
    info = {"capturas": list(capturas), "n_anuncios": int(len(df)), "limite_p99": limite,
            "localidades": localidades, "tipologias": tipologias}
    return df, info


def entrenar_paquete(usar_coordenadas: bool, df: pd.DataFrame | None = None,
                     info: dict | None = None, metricas_validacion: dict | None = None) -> dict:
    """Entrena los tres modelos con los datos finales y arma el paquete para la API."""
    if df is None or info is None:
        df, info = preparar_entrenamiento_final()
    version = version_de(usar_coordenadas)
    precio, q_inf, q_sup = _entrenar(df, version)
    return {
        # Claves compatibles con exportar.guardar_modelo
        "modelo": precio, "q_inf": q_inf, "q_sup": q_sup, "version": version,
        # Información para la API y el tablero
        "usar_coordenadas": usar_coordenadas,
        "columnas": variables_v2.columnas(version),
        "localidades": info["localidades"],
        "tipologias": info["tipologias"],
        "percentiles": PERCENTILES,
        "parametros_precio": PARAMETROS_PRECIO[version],
        "parametros_rango": PARAMETROS_RANGO,
        "metricas_validacion": metricas_validacion or {},
        "entrenamiento": {k: info[k] for k in ("capturas", "n_anuncios", "limite_p99")},
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # La API debe cargar el paquete con estas mismas versiones de las librerías
        "versiones": {"lightgbm": lgb.__version__, "scikit-learn": sklearn.__version__,
                      "pandas": pd.__version__, "numpy": np.__version__},
    }


def guardar_paquete(paquete: dict, ruta: Path | str | None = None) -> Path:
    ruta = Path(ruta or ruta_paquete(paquete["usar_coordenadas"]))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquete, ruta, compress=3)
    return ruta


def cargar_paquete(usar_coordenadas: bool = False, ruta: Path | str | None = None) -> dict:
    ruta = Path(ruta or ruta_paquete(usar_coordenadas))
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecute el cuaderno 09_modelo_final para generar los modelos.")
    return joblib.load(ruta)
