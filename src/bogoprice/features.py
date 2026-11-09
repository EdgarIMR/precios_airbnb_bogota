"""Ingeniería de características y preprocesamiento.

`construir_features` se usa igual en entrenamiento y en la API, para que el
modelo reciba siempre las mismas columnas.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# Patrón (minúsculas) que activa cada amenidad. Basado en los textos reales de la
# captura de junio 2026 (p. ej. "Wifi", "Free parking on premises", "Hot tub").
AMENIDADES = {
    "tiene_wifi": r"wifi",
    "tiene_parqueadero": r"parking",
    "tiene_ascensor": r"elevator",
    "tiene_zona_trabajo": r"workspace",
    "tiene_agua_caliente": r"^hot water$",
    "tiene_jacuzzi": r"hot tub",
}


def _numero_banos(texto) -> float:
    """'2.5 baths' -> 2.5, 'Shared half-bath' -> 0.5, vacío -> NaN."""
    if pd.isna(texto):
        return np.nan
    texto = str(texto).lower()
    numero = re.search(r"\d+(\.\d+)?", texto)
    if numero:
        return float(numero.group())
    return 0.5 if "half" in texto else np.nan


def _flags_amenidades(amenities: pd.Series) -> pd.DataFrame:
    listas = amenities.fillna("[]").map(
        lambda s: [a.lower() for a in json.loads(s)] if s.startswith("[") else []
    )
    return pd.DataFrame(
        {
            col: listas.map(lambda lst, p=patron: int(any(re.search(p, a) for a in lst)))
            for col, patron in AMENIDADES.items()
        },
        index=amenities.index,
    )


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega bathrooms_count, is_shared_bathroom y las amenidades binarias.

    Si el DataFrame ya trae esas columnas (caso API), se respetan.
    """
    out = df.copy()
    if "bathrooms_count" not in out and "bathrooms_text" in out:
        out["bathrooms_count"] = out["bathrooms_text"].map(_numero_banos)
    if "is_shared_bathroom" not in out and "bathrooms_text" in out:
        out["is_shared_bathroom"] = (
            out["bathrooms_text"].fillna("").str.lower().str.contains("shared").astype(int)
        )
    if "amenities" in out:
        faltantes = [c for c in AMENIDADES if c not in out]
        if faltantes:
            out = out.join(_flags_amenidades(out["amenities"])[faltantes])
    return out


def filtrar_localidades(df: pd.DataFrame, localidades: list[str]) -> pd.DataFrame:
    return df[df["neighbourhood_cleansed"].isin(localidades)].copy()


def localidades_habilitadas(df_entrenamiento: pd.DataFrame) -> list[str]:
    conteo = df_entrenamiento["neighbourhood_cleansed"].value_counts()
    return sorted(conteo[conteo >= config.MIN_ANUNCIOS_LOCALIDAD].index)


def construir_preprocesador(escalar: bool = True) -> ColumnTransformer:
    """One-hot para categóricas; imputación (y escala opcional) para numéricas."""
    numericas = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        numericas.append(("escalar", StandardScaler()))
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAS),
            ("num", Pipeline(numericas), config.NUMERICAS),
            ("bin", SimpleImputer(strategy="most_frequent"), config.BINARIAS),
        ]
    )


def construir_pipeline(modelo, escalar: bool = True) -> Pipeline:
    """Pipeline completo: preprocesamiento + modelo (entrena sobre log_price)."""
    return Pipeline([("prep", construir_preprocesador(escalar)), ("modelo", modelo)])
