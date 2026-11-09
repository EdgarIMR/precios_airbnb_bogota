"""Carga y limpieza de las capturas de Inside Airbnb.

Reproduce exactamente la lógica del cuaderno de EDA (notebooks/00_eda.ipynb):
- price de texto monetario a float (COP)
- se descartan precios <= 0 y los mayores al P99 de la captura de entrenamiento
- log_price = log(price)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def cargar_y_limpiar(captura_o_ruta) -> pd.DataFrame:
    """Lee una captura ('2026-06', '2026-09' o una ruta) y convierte price a float."""
    ruta = (
        config.ruta_captura(captura_o_ruta)
        if str(captura_o_ruta) in config.CAPTURAS
        else captura_o_ruta
    )
    df = pd.read_csv(ruta, compression="gzip", low_memory=False)
    df[config.TARGET] = pd.to_numeric(
        df[config.TARGET]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )
    return df


def limite_atipicos(df_entrenamiento: pd.DataFrame) -> float:
    """P99 del precio en la captura de entrenamiento (se reutiliza en evaluación)."""
    return float(df_entrenamiento[config.TARGET].quantile(config.PERCENTIL_ATIPICOS))


def filtrar_precio(df: pd.DataFrame, limite_alto: float) -> pd.DataFrame:
    """Deja precios en (0, limite_alto] y agrega log_price."""
    precio = df[config.TARGET]
    out = df[(precio > 0) & (precio <= limite_alto)].copy()
    out[config.TARGET_LOG] = np.log(out[config.TARGET])
    return out


def cargar_train_test() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Entrenamiento = junio, evaluación = septiembre, con el mismo límite P99."""
    junio = cargar_y_limpiar(config.CAPTURA_ENTRENAMIENTO)
    septiembre = cargar_y_limpiar(config.CAPTURA_EVALUACION)
    limite = limite_atipicos(junio)
    return filtrar_precio(junio, limite), filtrar_precio(septiembre, limite), limite
