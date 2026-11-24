"""Preparación de datos y evaluación de modelos"""
from __future__ import annotations

 
import numpy as np
import pandas as pd

from . import config, variables_v2
from .data_loader import cargar_train_test, cargar_y_limpiar
from .features import construir_features, filtrar_localidades, localidades_habilitadas
from .metrics import mdape, metricas_precio
from .tracking import registrar_corrida

# Se realiza validación cruzada para elegir los hiperparámetros con los datos de junio y los
# anuncios nuevos de septiembre quedan como prueba, que se usa una sola vez al final.
N_PLIEGUES = 10
SEMILLA_PLIEGUES = 17

#EXPERIMENTO_MODELOS = "bogoprice-modelos"


class Datos:
    def __init__(self, train, test, es_nuevo, localidades, tipologias, limite_p99):
        self.train = train
        self.test = test
        self.es_nuevo = es_nuevo              # anuncios de septiembre que no existían en junio
        self.localidades = localidades
        self.tipologias = tipologias
        self.limite_p99 = limite_p99
        # Log del precio en junio y precio en COP de septiembre
        self.y_train = train[config.TARGET_LOG]
        self.y_test_cop = test[config.TARGET].values


def preparar_datos() -> Datos:
    """ Se usa junio como entrenamiento y septiembre para evaluación."""
    train, test, limite = cargar_train_test()
    train, test = construir_features(train), construir_features(test)
    localidades = localidades_habilitadas(train)
    train, test = filtrar_localidades(train, localidades), filtrar_localidades(test, localidades)
    tipologias = variables_v2.tipologias_frecuentes(train)
    train = variables_v2.agregar_variables_v2(train, tipologias)
    test = variables_v2.agregar_variables_v2(test, tipologias)
    ids_junio = set(cargar_y_limpiar(config.CAPTURA_ENTRENAMIENTO)["id"])
    es_nuevo = ~test["id"].isin(ids_junio).values
    print(f"P99 junio: ${limite:,.0f} | train {len(train)} | test {len(test)} | "
          f"nuevos {es_nuevo.sum()} | localidades {len(localidades)} | tipologías {len(tipologias)}")
    return Datos(train, test, es_nuevo, localidades, tipologias, limite)

# Evaluación fuera de tiempo con entrenamiento en junio y validacion en septiembre
def evaluar(datos: Datos, nombre: str, modelo, version: str, params: dict, autor: str,
            escalar: bool = False, denso: bool = True, registrar: bool = True):
    """Entrena sobre log(price), evalúa en COP."""
    cols = variables_v2.columnas(version)
    pipe = variables_v2.construir_pipeline(modelo, version, escalar, denso)
    pipe.fit(datos.train[cols], datos.y_train)
    pred = np.exp(pipe.predict(datos.test[cols]))
    m = metricas_precio(datos.y_test_cop, pred)
    m["mdape_nuevos"] = mdape(datos.y_test_cop[datos.es_nuevo], pred[datos.es_nuevo])
    m["mdape_train"] = mdape(datos.train[config.TARGET], np.exp(pipe.predict(datos.train[cols])))
    if registrar:
        registrar_corrida(
            nombre, {**params, "conjunto": version, "n_features": len(cols)}, m, pipeline=pipe,
            tags={"autor": autor, "conjunto": version, "modelo": params.get("modelo", ""),
                  "features": ",".join(cols)},
        )
    return m, pipe


def tabla_resultados(resultados: list[dict]) -> pd.DataFrame:
    cols = ["corrida", "conjunto", "mdape", "mdape_nuevos", "mdape_train", "mae_cop"]
    df = pd.DataFrame(resultados)
    return df[[c for c in cols if c in df]].sort_values("mdape_nuevos").round(2)


#  Validación cruzada sobre junio  para seleccionar hiperparámetros
def pliegues_cv(datos: Datos, n_pliegues: int = N_PLIEGUES,
                semilla: int = SEMILLA_PLIEGUES) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pliegues de junio agrupados por anfitrión y estratificados por localidad.

    - Agrupados (host_id): Un anfitrión no puede tener anuncios en pliegues separados para evitar que el modelo se sesgue por anfitrión.
    - Estratificados (neighbourhood_cleansed): Se tiene que la muestra de los pliegues representa de forma cercana la población de localidades que se tiene en junio.
    - Se usa una semilla para que se pueda comparar entre modelos.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=n_pliegues, shuffle=True, random_state=semilla)
    estratos = datos.train["neighbourhood_cleansed"].values
    grupos = datos.train["host_id"].values
    return list(cv.split(datos.train, estratos, grupos))


def resumen_pliegues(datos: Datos, pliegues) -> pd.DataFrame:
    """Tamaño de cada pliegue."""
    filas = []
    for i, (idx_ent, idx_val) in enumerate(pliegues, start=1):
        val = datos.train.iloc[idx_val]
        filas.append({"pliegue": i, "n_entrenamiento": len(idx_ent), "n_validacion": len(idx_val),
                      "pct_validacion": 100 * len(idx_val) / len(datos.train),
                      "anfitriones_validacion": val["host_id"].nunique(),
                      "precio_mediano_validacion": val[config.TARGET].median()})
    return pd.DataFrame(filas)


def evaluar_cv(datos: Datos, pliegues, nombre: str, modelo, version: str, params: dict,
               autor: str, escalar: bool = False,
               denso: bool = True, registrar: bool = True) -> dict:
    """Validación cruzada sobre junio.

    Por cada pliegue entrena con el resto de junio y mide en COP sobre el pliegue que queda
    fuera. 
    Devuelve el promedio y la desviación estándar entre pliegues.
    """
    from sklearn.base import clone

    cols = variables_v2.columnas(version)
    base = variables_v2.construir_pipeline(modelo, version, escalar, denso)
    X, y_log = datos.train[cols], datos.y_train.values
    y_cop = datos.train[config.TARGET].values

    por_pliegue = []
    for idx_ent, idx_val in pliegues:
        pipe = clone(base)
        pipe.fit(X.iloc[idx_ent], y_log[idx_ent])
        pred_val = np.exp(pipe.predict(X.iloc[idx_val]))
        pred_ent = np.exp(pipe.predict(X.iloc[idx_ent]))
        m = metricas_precio(y_cop[idx_val], pred_val)
        m["mdape_train"] = mdape(y_cop[idx_ent], pred_ent)
        por_pliegue.append(m)

    tabla = pd.DataFrame(por_pliegue)
    resultado = {
        "mdape_cv": tabla["mdape"].mean(),
        "mdape_cv_std": tabla["mdape"].std(ddof=1),
        "mae_cv": tabla["mae_cop"].mean(),
        "rmse_log_cv": tabla["rmse_log"].mean(),
        "mdape_train_cv": tabla["mdape_train"].mean(),
    }
    resultado["brecha_cv"] = resultado["mdape_cv"] - resultado["mdape_train_cv"]

    if registrar:
        metricas_mlflow = dict(resultado)
        for i, mdape_i in enumerate(tabla["mdape"], start=1):
            metricas_mlflow[f"mdape_pliegue_{i:02d}"] = mdape_i
        registrar_corrida(
            nombre,
            {**params, "conjunto": version, "n_features": len(cols),
             "validacion": f"StratifiedGroupKFold(k={len(pliegues)}, grupo=host_id, "
                           f"estrato=localidad, semilla={SEMILLA_PLIEGUES})"},
            metricas_mlflow,
            tags={"autor": autor, "conjunto": version, "modelo": params.get("modelo", ""),
                  "etapa": "validacion_cruzada", "features": ",".join(cols)},
        )
    return resultado


def evaluar_prueba(datos: Datos, nombre: str, modelo, version: str, params: dict, autor: str,
                   escalar: bool = False, denso: bool = True, registrar: bool = True):
    """Evaluación final : entrena con todo junio y mide
    sobre septiembre. Devuelve las métricas y el modelo entrenado"""
    cols = variables_v2.columnas(version)
    pipe = variables_v2.construir_pipeline(modelo, version, escalar, denso)
    pipe.fit(datos.train[cols], datos.y_train)
    pred = np.exp(pipe.predict(datos.test[cols]))
    y = datos.y_test_cop
    m = metricas_precio(y, pred)
    m_nuevos = metricas_precio(y[datos.es_nuevo], pred[datos.es_nuevo])
    m.update({"mdape_nuevos": m_nuevos["mdape"], "mae_nuevos": m_nuevos["mae_cop"],
              "rmse_log_nuevos": m_nuevos["rmse_log"]})
    m["mdape_train"] = mdape(datos.train[config.TARGET], np.exp(pipe.predict(datos.train[cols])))
    if registrar:
        registrar_corrida(
            nombre, {**params, "conjunto": version, "n_features": len(cols)}, m, pipeline=pipe,
            tags={"autor": autor, "conjunto": version, "modelo": params.get("modelo", ""),
                  "etapa": "prueba_final", "features": ",".join(cols)},
        )
    return m, pipe


def intervalo_bootstrap(y_real, y_pred, n: int = 1000, semilla: int = 0,
                        nivel: float = 0.95) -> tuple[float, float]:
    """Intervalo de confianza por bootstrap del MdAPE (en %), para reportar la
    incertidumbre del resultado en los 1.632 anuncios nuevos."""
    rng = np.random.default_rng(semilla)
    y_real, y_pred = np.asarray(y_real, float), np.asarray(y_pred, float)
    errores = np.abs(y_pred - y_real) / y_real
    muestras = [np.median(errores[rng.integers(0, len(errores), len(errores))]) * 100
                for _ in range(n)]
    a = (1 - nivel) / 2
    return float(np.quantile(muestras, a)), float(np.quantile(muestras, 1 - a))
