"""Atajos para MLflow, de modo que todos los cuadernos registren igual."""
from __future__ import annotations

import mlflow

from . import config


def iniciar_mlflow(experimento: str) -> None:
    """Apunta al servidor MLflow de EC2 (MLFLOW_TRACKING_URI en .env) y fija el experimento."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experimento)


def registrar_corrida(nombre: str, params: dict, metricas: dict, pipeline=None,
                      artefactos: list[str] | None = None, tags: dict | None = None) -> str:
    """Crea una corrida con parámetros, métricas, modelo y artefactos. Devuelve el run_id."""
    with mlflow.start_run(run_name=nombre) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metricas)
        mlflow.set_tags({"features": ",".join(config.FEATURES), **(tags or {})})
        if pipeline is not None:
            # Argumento posicional: funciona en MLflow 2.x y 3.x.
            # cloudpickle: las versiones recientes usan 'skops' por defecto y
            # rechazan los árboles de sklearn como "tipos no confiables".
            mlflow.sklearn.log_model(
                pipeline,
                "model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
        for ruta in artefactos or []:
            mlflow.log_artifact(ruta)
        return run.info.run_id
