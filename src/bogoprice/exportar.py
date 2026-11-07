"""Exporta el modelo elegido al formato que carga la API (models/modelo.joblib)."""
from __future__ import annotations

from pathlib import Path

import joblib

from . import config


def guardar_modelo(modelo, version: str, q_inf=None, q_sup=None,
                   ruta: Path | None = None) -> Path:
    """Guarda {modelo, q_inf, q_sup, version}. q_inf/q_sup: pipelines de cuantil 0.1 y 0.9."""
    ruta = Path(ruta or config.MODELO_PATH)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "q_inf": q_inf, "q_sup": q_sup, "version": version}, ruta)
    return ruta
