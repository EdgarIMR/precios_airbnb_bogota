"""Rutas de la API: salud, opciones del formulario, predicción y explicación."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app import config
from app.schemas import Anuncio, Explicacion, Health, Opciones, Prediccion
from predictor import Predictor

api_router = APIRouter()
predictor = Predictor(config.MODELOS_DIR)


@api_router.get("/health", response_model=Health, status_code=200)
def health() -> dict:
    modelos = []
    for geo in (False, True):
        try:
            modelos.append(f"LightGBM {predictor.paquete(geo)['version']}")
        except FileNotFoundError:
            pass
    return {"name": config.PROJECT_NAME, "api_version": config.VERSION_API, "modelos": modelos}


@api_router.get("/opciones", response_model=Opciones, status_code=200)
def opciones(usar_coordenadas: bool = Query(False)) -> dict:
    o = predictor.opciones(usar_coordenadas)
    if o["limites_coordenadas"]:
        o["limites_coordenadas"] = {k: list(v) for k, v in o["limites_coordenadas"].items()}
    o["metricas_validacion"] = {k: float(v) for k, v in o["metricas_validacion"].items()}
    return o


@api_router.post("/predict", response_model=Prediccion, status_code=200)
def predict(anuncio: Anuncio) -> dict:
    try:
        return predictor.predecir(anuncio.crudo(), usar_coordenadas=anuncio.usar_coordenadas)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@api_router.post("/explicar", response_model=Explicacion, status_code=200)
def explicar(anuncio: Anuncio) -> dict:
    try:
        tabla = predictor.explicar(anuncio.crudo(), usar_coordenadas=anuncio.usar_coordenadas)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"precio_base_cop": tabla.attrs["precio_base_cop"],
            "precio_referencia_cop": tabla.attrs["precio_referencia_cop"],
            "aportes": tabla.to_dict(orient="records")}
