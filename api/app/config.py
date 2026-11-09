"""Configuración de la API de BogoPrice."""
import os
from pathlib import Path

PROJECT_NAME = "BogoPrice API"
API_V1_STR = "/api/v1"
VERSION_API = "1.0.0"
# Carpeta con modelo_v2.joblib y modelo_v2_geo.joblib (se puede cambiar con MODELOS_DIR)
MODELOS_DIR = Path(os.getenv("MODELOS_DIR", Path(__file__).resolve().parents[1] / "modelos"))
