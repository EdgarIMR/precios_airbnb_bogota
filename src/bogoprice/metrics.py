"""Métricas de éxito del proyecto (Tabla 1 del informe 1).

Todas se calculan en COP, es decir, después de deshacer el log.
"""
from __future__ import annotations

import numpy as np


def mdape(y_real, y_pred) -> float:
    """Error porcentual absoluto mediano, en %. Meta: <= 20 %."""
    y_real, y_pred = np.asarray(y_real, float), np.asarray(y_pred, float)
    return float(np.median(np.abs(y_pred - y_real) / y_real) * 100)


def mae(y_real, y_pred) -> float:
    y_real, y_pred = np.asarray(y_real, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_pred - y_real)))


def cobertura(y_real, limite_inf, limite_sup) -> float:
    """% de precios reales dentro del rango estimado. Meta: ~80 %."""
    y_real = np.asarray(y_real, float)
    dentro = (y_real >= np.asarray(limite_inf, float)) & (y_real <= np.asarray(limite_sup, float))
    return float(dentro.mean() * 100)


def metricas_precio(y_real_cop, y_pred_cop) -> dict[str, float]:
    """Diccionario listo para mlflow.log_metrics."""
    y_real_cop = np.asarray(y_real_cop, float)
    y_pred_cop = np.asarray(y_pred_cop, float)
    return {
        "mdape": mdape(y_real_cop, y_pred_cop),
        "mae_cop": mae(y_real_cop, y_pred_cop),
        "rmse_log": float(np.sqrt(np.mean((np.log(y_pred_cop) - np.log(y_real_cop)) ** 2))),
    }
