"""Preparación de datos y evaluación de modelos para los cuadernos 03 a 08.

Centraliza el código común para que todos los modelos se entrenen y evalúen igual.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, variables_v2
from .data_loader import cargar_train_test, cargar_y_limpiar
from .features import construir_features, filtrar_localidades, localidades_habilitadas
from .metrics import mdape, metricas_precio
from .tracking import registrar_corrida

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


def evaluar(datos: Datos, nombre: str, modelo, version: str, params: dict, autor: str,
            escalar: bool = False, denso: bool = True, registrar: bool = True):
    """Entrena sobre log(price), evalúa en COP (toda la captura y anuncios nuevos)
    y registra la corrida en MLflow. Devuelve (métricas, pipeline)."""
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
