"""Acceso al modelo desde el tablero.

Semana 5: el tablero carga models/modelo.joblib directamente.
Semana 6: si se define la variable de entorno API_URL, el tablero llama a la API
(POST {API_URL}/predict) y deja de cargar el modelo localmente. Así el código del
tablero no cambia cuando se despliegue la API.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from bogoprice import config

_MODELO = None


def _cargar():
    global _MODELO
    if _MODELO is None:
        _MODELO = joblib.load(config.MODELO_PATH)
    return _MODELO


def version_modelo() -> str:
    if os.getenv("API_URL"):
        return "API " + os.environ["API_URL"]
    return _cargar()["version"]


def predecir(anuncio: dict) -> dict:
    """Recibe las variables de config.FEATURES y devuelve precio y rango en COP."""
    if os.getenv("API_URL"):
        import requests

        r = requests.post(f"{os.environ['API_URL']}/predict", json=anuncio, timeout=10)
        r.raise_for_status()
        return r.json()

    paquete = _cargar()
    X = pd.DataFrame([anuncio])[config.FEATURES]
    precio = float(np.exp(paquete["modelo"].predict(X)[0]))
    salida = {"precio": precio, "inferior": None, "superior": None}
    if paquete.get("q_inf") is not None:
        inf = float(np.exp(paquete["q_inf"].predict(X)[0]))
        sup = float(np.exp(paquete["q_sup"].predict(X)[0]))
        # Los cuantiles se estiman por separado: se ordenan y se asegura que
        # el rango contenga el precio de referencia.
        salida["inferior"] = min(inf, sup, precio)
        salida["superior"] = max(inf, sup, precio)
    return salida
