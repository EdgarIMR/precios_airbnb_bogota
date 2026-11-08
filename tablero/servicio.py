"""Acceso al modelo desde el tablero.

Semana 5: el tablero carga models/modelo.joblib directamente.
Semana 6: si se define la variable de entorno API_URL, el tablero llama a la API
(POST {API_URL}/predict) para el precio y el rango, sin cambiar el resto del código.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from bogoprice import config

_MODELO = None

# Nombre visible de cada variable original del modelo
NOMBRES = {
    "neighbourhood_cleansed": "Localidad",
    "room_type": "Tipo de alojamiento",
    "accommodates": "Capacidad",
    "bedrooms": "Habitaciones",
    "beds": "Camas",
    "bathrooms_count": "Baños",
    "minimum_nights": "Noches mínimas",
    "is_shared_bathroom": "Baño compartido",
    "tiene_wifi": "Wifi",
    "tiene_parqueadero": "Parqueadero",
    "tiene_ascensor": "Ascensor",
    "tiene_zona_trabajo": "Zona de trabajo",
    "tiene_agua_caliente": "Agua caliente",
    "tiene_jacuzzi": "Jacuzzi",
}


def _cargar():
    global _MODELO
    if _MODELO is None:
        _MODELO = joblib.load(config.MODELO_PATH)
    return _MODELO


def version_modelo() -> str:
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
        # Los cuantiles se estiman por separado: se ordenan y el rango debe
        # contener el precio de referencia.
        salida["inferior"] = min(inf, sup, precio)
        salida["superior"] = max(inf, sup, precio)
    return salida


def contribuciones(anuncio: dict) -> pd.DataFrame:
    """Contribución de cada variable al precio estimado, en COP.

    El modelo es lineal en log(precio): log p = b0 + sum_j beta_j * z_j.
    Para la variable original k (que puede ocupar varias columnas z después del
    one-hot) se calcula d_k = sum_{j en k} beta_j * (z_j - media_j), su efecto
    frente a un alojamiento promedio de entrenamiento. En COP:
        contribución_k = p - p * exp(-d_k)
    es decir, cuánto bajaría (o subiría) el precio si esa variable tomara el
    valor promedio y las demás se quedaran igual.
    """
    paquete = _cargar()
    pipe = paquete["modelo"]
    prep, reg = pipe.named_steps["prep"], pipe.named_steps["modelo"]
    X = pd.DataFrame([anuncio])[config.FEATURES]
    z = np.asarray(prep.transform(X)).ravel()
    d = reg.coef_ * (z - paquete["medias_transformadas"])
    precio = float(np.exp(pipe.predict(X)[0]))

    nombres_z = prep.get_feature_names_out()
    filas = []
    for var in config.FEATURES:
        mascara = np.array([n.split("__", 1)[1].startswith(var) for n in nombres_z])
        dk = float(d[mascara].sum())
        filas.append({"variable": NOMBRES.get(var, var), "cop": precio - precio * np.exp(-dk)})
    return pd.DataFrame(filas)
