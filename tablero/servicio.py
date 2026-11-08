"""Acceso al modelo desde el tablero.

Si existe la variable de entorno API_URL (por ejemplo http://localhost:8001), el tablero
consulta la API de BogoPrice (POST /api/v1/predict y /api/v1/explicar). Si no, carga el
mismo predictor de la API desde la carpeta api/ del repositorio. En los dos casos el
modelo y la transformación de variables son los mismos.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

API_URL = os.getenv("API_URL", "").rstrip("/")
CARPETA_API = Path(__file__).resolve().parents[1] / "api"


@lru_cache(maxsize=1)
def _predictor():
    sys.path.insert(0, str(CARPETA_API))
    from predictor import Predictor

    return Predictor()


def _post(ruta: str, anuncio: dict, usar_coordenadas: bool) -> dict:
    import requests

    r = requests.post(f"{API_URL}/api/v1/{ruta}", json={**anuncio, "usar_coordenadas": usar_coordenadas},
                      timeout=15)
    if r.status_code == 422:
        raise ValueError(r.json().get("detail"))
    r.raise_for_status()
    return r.json()


def origen() -> str:
    return f"API {API_URL}" if API_URL else "predictor local (api/predictor.py)"


def opciones() -> dict:
    if API_URL:
        import requests

        return requests.get(f"{API_URL}/api/v1/opciones", timeout=15).json()
    return _predictor().opciones(False)


def predecir(anuncio: dict, usar_coordenadas: bool = False) -> dict:
    """Devuelve precio de referencia, límites del rango y nombre del modelo."""
    p = (_post("predict", anuncio, usar_coordenadas) if API_URL
         else _predictor().predecir(anuncio, usar_coordenadas=usar_coordenadas))
    return {"precio": p["precio_referencia"], "inferior": p["rango_inferior"],
            "superior": p["rango_superior"], "modelo": p["modelo"]}


def contribuciones(anuncio: dict, usar_coordenadas: bool = False) -> pd.DataFrame:
    """Aporte de cada variable al precio, en pesos (columnas variable y cop)."""
    if API_URL:
        tabla = pd.DataFrame(_post("explicar", anuncio, usar_coordenadas)["aportes"])
    else:
        tabla = _predictor().explicar(anuncio, usar_coordenadas=usar_coordenadas)
    return tabla.rename(columns={"efecto_cop": "cop"})[["variable", "cop"]]
