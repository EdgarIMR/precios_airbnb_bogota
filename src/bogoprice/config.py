"""Configuración central del proyecto BogoPrice.

Todas las rutas y listas de columnas se definen aquí para que los cuadernos,
la API y el tablero usen exactamente lo mismo.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Rutas -----------------------------------------------------------------
# Raíz del repositorio (src/bogoprice/config.py -> ../../)
ROOT = Path(__file__).resolve().parents[2]

# Carga variables de .env si python-dotenv está instalado (opcional)
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # Lo administra DVC (data/raw.dvc)
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Nombres de las capturas de Inside Airbnb. Ver docs/datos.md.
# Se busca primero en data/raw/ (lo que deja `dvc pull`) y luego en las copias que hay
# en Git en data/ (listings0626.csv.gz). Camino por defecto: las copias de Git.
# OJO: los nombres dentro de data/raw/ NO están verificados; si `ls data/raw` muestra otros
# nombres, ajustar la primera ruta de cada captura.
CAPTURAS = {
    "2026-06": ["raw/listings_2026-06.csv.gz", "listings0626.csv.gz"],
    "2026-09": ["raw/listings_2026-09.csv.gz", "listings0926.csv.gz"],
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
# PENDIENTE: el informe 1 dice que se fija con la exploración (la maqueta usa 500).
MIN_ANUNCIOS_LOCALIDAD = int(os.getenv("MIN_ANUNCIOS_LOCALIDAD", "500"))

# --- Variables del modelo servible -----------------------------------------
# Solo lo que el anfitrión ingresa en el formulario de la maqueta.
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

# Variables extra solo para experimentos de referencia (NO servibles: un anuncio
# nuevo no tiene reseñas ni historial de anfitrión).
FEATURES_REFERENCIA = [
    "host_is_superhost",
    "host_identity_verified",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_location",
    "instant_bookable",
]

# --- MLflow ------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODELO_REGISTRADO = "bogoprice"

# --- Servicio ----------------------------------------------------------------
MODELO_PATH = Path(os.getenv("MODELO_PATH", MODELS_DIR / "modelo.joblib"))
API_URL = os.getenv("API_URL", "http://localhost:8000")
