#!/bin/bash
# Arranca la API en el puerto 8001 (el mismo del taller de APIs)
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8001}"
