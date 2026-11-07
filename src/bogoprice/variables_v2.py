"""Variables v2  — complementan las de config.py y features.py.

Agrega más amenidades, la tipología agrupada, las noches mínimas recortadas, la estancia mensual y define tres conjuntos de variables:

    v1      variables de config.FEATURES (cuaderno 02)
    v2      v1 + tipología, 10 amenidades más, noches mínimas recortadas y estancia mensual
    v2_geo  v2 + latitud y longitud

"""
from __future__ import annotations

import json
import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# --- Parámetros -------------------------------------------------------------------
# Cantidad minima de tipo de inmueble que debe tener para conservarlo el resto se maneja como "Other"
MIN_ANUNCIOS_TIPOLOGIA = 50
# Se trabaja como maximo el "30 o mas"     
MAX_NOCHES_MINIMAS = 30         
# Airbnb aplica el descuento mensual desde 28 noches
UMBRAL_ESTANCIA_MENSUAL = 28    

# Amenidades adicionales (patrones en minúsculas sobre los textos de Inside Airbnb)
AMENIDADES_V2 = {
    "tiene_cocina": r"^kitchen$",
    "tiene_lavadora": r"washer",
    "tiene_aire_acondicionado": r"air conditioning|^ac ",
    "tiene_piscina": r"pool",
    "tiene_gimnasio": r"gym",
    "tiene_vista": r"view",
    "tiene_balcon": r"patio or balcony",
    "tiene_tv": r"^tv|hdtv",
    "tiene_autocheckin": r"self check-in",
    "acepta_mascotas": r"pets allowed",
}

# --- Conjuntos de variables -------------------------------------------------------
CATEGORICAS_V2 = config.CATEGORICAS + ["tipologia"]
NUMERICAS_V2 = ["accommodates", "bedrooms", "beds", "bathrooms_count", "noches_minimas"]
BINARIAS_V2 = config.BINARIAS + list(AMENIDADES_V2) + ["estancia_mensual"]
GEO = ["latitude", "longitude"]

CONJUNTOS = {
    "v1": (config.CATEGORICAS, config.NUMERICAS, config.BINARIAS),
    "v2": (CATEGORICAS_V2, NUMERICAS_V2, BINARIAS_V2),
    "v2_geo": (CATEGORICAS_V2, NUMERICAS_V2 + GEO, BINARIAS_V2),
}


def columnas(version: str = "v1") -> list[str]:
    """Columnas de entrada del modelo para un conjunto de variables."""
    categoricas, numericas, binarias = CONJUNTOS[version]
    return categoricas + numericas + binarias


# --- Construcción de variables ------------------------------------------------------
def tipologias_frecuentes(df_entrenamiento: pd.DataFrame) -> list[str]:
    """Tipologías (property_type) con suficientes anuncios en entrenamiento (junio)."""
    conteo = df_entrenamiento["property_type"].value_counts()
    return sorted(conteo[conteo >= MIN_ANUNCIOS_TIPOLOGIA].index)


def agregar_variables_v2(df: pd.DataFrame, tipologias: list[str]) -> pd.DataFrame:
    """Agrega las variables v2. La lista de tipologías se calcula con junio y se
    reutiliza en septiembre y en la API. Si una columna ya existe (caso API), se respeta."""
    out = df.copy()

    if "amenities" in out:
        faltantes = [c for c in AMENIDADES_V2 if c not in out]
        if faltantes:
            listas = out["amenities"].fillna("[]").map(
                lambda s: [a.lower() for a in json.loads(s)] if s.startswith("[") else []
            )
            for columna in faltantes:
                patron = re.compile(AMENIDADES_V2[columna])
                out[columna] = listas.map(lambda lst: int(any(patron.search(a) for a in lst)))

    # Noches mínimas se usa la  moda (1) en vacíos y recorte a [1, 30]
    if "noches_minimas" not in out and "minimum_nights" in out:
        out["noches_minimas"] = out["minimum_nights"].fillna(1).clip(1, MAX_NOCHES_MINIMAS)
    if "estancia_mensual" not in out and "noches_minimas" in out:
        out["estancia_mensual"] = (out["noches_minimas"] >= UMBRAL_ESTANCIA_MENSUAL).astype(int)

    # Tipología agrupada frecuente se conserva el resto se pasa a "Other"
    if "tipologia" not in out and "property_type" in out:
        out["tipologia"] = out["property_type"].where(out["property_type"].isin(tipologias), "Other")
    return out


# --- Preprocesamiento -----------------------------------------------------------------
def construir_preprocesador(version: str = "v2", escalar: bool = False,
                            denso: bool = True) -> ColumnTransformer:
    """Igual al de features.py pero para cualquier conjunto de variables.
    denso=True: matriz densa (la requiere CatBoost)."""
    categoricas, numericas, binarias = CONJUNTOS[version]
    pasos_num = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        pasos_num.append(("escalar", StandardScaler()))
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=not denso), categoricas),
            ("num", Pipeline(pasos_num), numericas),
            ("bin", SimpleImputer(strategy="most_frequent"), binarias),
        ],
        sparse_threshold=0 if denso else 0.3,
    )


def construir_pipeline(modelo, version: str = "v2", escalar: bool = False,
                       denso: bool = True) -> Pipeline:
    """Preprocesamiento + modelo (se entrena sobre log_price)."""
    return Pipeline([("prep", construir_preprocesador(version, escalar, denso)), ("modelo", modelo)])
