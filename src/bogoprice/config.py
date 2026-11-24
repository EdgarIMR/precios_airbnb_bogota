"""Configuración central del proyecto BogoPrice.

Rutas, capturas de datos y listas de columnas compartidas por los cuadernos y el
entrenamiento del modelo final. La API (api/) y el tablero (tablero/) no dependen de
este paquete: el modelo empaquetado trae todo lo que necesitan.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Rutas -----------------------------------------------------------------
# Raíz del repositorio (src/bogoprice/config.py -> ../../)
ROOT = Path(__file__).resolve().parents[2]

# Carga variables de .env si python-dotenv está instalado (ver .env.example)
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DATA_DIR = ROOT / "data"

# Capturas de Inside Airbnb Bogotá, versionadas con DVC (data/*.csv.gz.dvc, remoto S3 en
# .dvc/config). Si no están en data/, se descargan con `dvc pull`.
CAPTURAS = {
    "2026-06": ["listings0626.csv.gz"],
    "2026-09": ["listings0926.csv.gz"],
}
CAPTURA_ENTRENAMIENTO = "2026-06"
CAPTURA_EVALUACION = "2026-09"


def ruta_captura(captura: str) -> Path:
    """Devuelve la ruta del archivo de una captura ('2026-06' o '2026-09')."""
    if captura not in CAPTURAS:
        raise KeyError(f"Captura desconocida: {captura}. Opciones: {list(CAPTURAS)}")
    for relativa in CAPTURAS[captura]:
        ruta = DATA_DIR / relativa
        if ruta.exists():
            return ruta
    raise FileNotFoundError(
        f"No se encontró la captura {captura}. Ejecuta `dvc pull` desde la raíz del repo. "
        f"Rutas buscadas: {[str(DATA_DIR / r) for r in CAPTURAS[captura]]}"
    )


# --- Variable objetivo y limpieza ----------------------------------------------
TARGET = "price"
TARGET_LOG = "log_price"
PERCENTIL_ATIPICOS = 0.99   # Mismo criterio del EDA (Entrega 1)

# Localidades habilitadas: mínimo de anuncios en la captura de entrenamiento.
# Con 500 quedan las 9 localidades con 700 o más anuncios (de Candelaria, 700, a
# Chapinero, 5.097); la siguiente, Kennedy, tiene 313 (ver cuaderno 01).
MIN_ANUNCIOS_LOCALIDAD = int(os.getenv("MIN_ANUNCIOS_LOCALIDAD", "500"))

# --- Variables del conjunto v1 (cuaderno 02) --------------------------------
# Solo lo que el anfitrión ingresa en el formulario. variables_v2.py las amplía.
CATEGORICAS = ["neighbourhood_cleansed", "room_type"]
NUMERICAS = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms_count",
    "minimum_nights",
]
BINARIAS = [
    "is_shared_bathroom",
    "tiene_wifi",
    "tiene_parqueadero",
    "tiene_ascensor",
    "tiene_zona_trabajo",
    "tiene_agua_caliente",
    "tiene_jacuzzi",
]
FEATURES = CATEGORICAS + NUMERICAS + BINARIAS

# --- MLflow ------------------------------------------------------------------
# Servidor de seguimiento en EC2 (se define en .env; ver .env.example)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
