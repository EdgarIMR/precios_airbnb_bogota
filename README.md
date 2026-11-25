# 🏢 Predicción de Precios de Referencia para Airbnb en Bogotá

> **Curso:** Despliegue de Soluciones Analíticas (MIAD) - Universidad de los Andes
>
> **Entrega:** Entrega final - Modelos, API y tablero desplegados en contenedores Docker
>
> **Dataset:** Capturas de Inside Airbnb Bogotá (Junio y Septiembre de 2026)

---

## 📌 Descripción del Proyecto

Este proyecto aborda la brecha de información que enfrentan los pequeños anfitriones independientes de Airbnb en Bogotá al definir las tarifas por noche de sus inmuebles.

Para reducir esta desventaja frente a operadores profesionales, se desarrolla una solución analítica de aprendizaje supervisado que estima un **precio de referencia por noche** y un **rango competitivo**, considerando las características físicas, geográficas y de estadía del alojamiento.

La solución se despliega mediante una **API de inferencia** y un **tablero interactivo** empaquetados en contenedores **Docker**.

---

## 👥 Integrantes del Equipo (Grupo 9)

- Edgar Iván Morales Rojas
  - e.moralesr@uniandes.edu.co

- Juan Camilo Piñeros Guzmán
  - j.pineros16@uniandes.edu.co

- Juan Esteban Carrero Herrera
  - je.carrero@uniandes.edu.co

- Manuel Guillermo González Panesso
  - mg.gonzalezp@uniandes.edu.co

---

## 📂 Estructura del Repositorio

```text
precios_airbnb_bogota/
├── docker-compose.yml        ← despliegue conjunto de la API y el tablero
├── pyproject.toml            ← paquete de entrenamiento `bogoprice` (pip install -e .)
├── requirements-modelos.txt  ← librerías adicionales de los cuadernos (Jupyter, CatBoost, DVC...)
├── .env.example              ← plantilla de variables (MLflow); copiar como .env
├── data/                     ← capturas de Inside Airbnb (versionadas con DVC en S3: data/*.csv.gz.dvc)
├── notebooks/                ← EDA, variables, experimentos y modelo final
├── src/bogoprice/            ← pipeline de datos, variables, métricas, MLflow y modelo final
├── api/                      ← API FastAPI + predictor + modelos empaquetados + Dockerfile
└── tablero/                  ← tablero Dash + insumos + Dockerfile
```

| Cuaderno | Contenido | Experimento(s) MLflow |
|---|---|---|
| `00_eda` | Exploración de las capturas de junio y septiembre, atípicos (P99) y figuras del informe | – |
| `01_features` | Variables servibles y umbral de localidades (500 anuncios) | – |
| `02_baseline_lineales` | Mediana por localidad × tipo, Ridge y ElasticNet (conjunto v1) | `bogoprice-baseline`, `bogoprice-lineales`, `bogoprice-lineales-features` |
| `03_random_forest` | Random Forest con los conjuntos v1, v2 y v2_geo | `bogoprice-random-forest*` |
| `04_lightgbm` | **Modelo elegido**: LightGBM, validación cruzada de 10 pliegues y prueba final | `bogoprice-lightgbm-cv`, `-ajuste-cv`, `-ajuste-fino-cv`, `bogoprice-prueba-final` |
| `06_catboost` | CatBoost con los mismos conjuntos de variables | `bogoprice-catboost` |
| `07_intervalos` | Rango sugerido con regresión cuantílica (percentiles 7,5–92,5) | `bogoprice-intervalos` |
| `09_modelo_final` | Entrena con junio + septiembre y exporta `api/modelos/*.joblib` | `bogoprice-modelo-final` |

| Módulo de `src/bogoprice/` | Qué hace |
|---|---|
| `config.py` | Rutas, capturas, variables v1 y URI de MLflow |
| `data_loader.py` | Lectura de capturas, precio a número y filtro de atípicos (P99 de junio) |
| `features.py`, `variables_v2.py` | Ingeniería de variables (baños, amenidades, tipología, noches mínimas) y preprocesamiento |
| `metrics.py` | MdAPE, MAE, RMSE (log) y cobertura del rango |
| `experimentos.py` | Preparación de datos, validación cruzada, prueba final e intervalo bootstrap |
| `tracking.py` | Registro de corridas en MLflow |
| `modelo_final.py` | Hiperparámetros elegidos y entrenamiento de los paquetes de la API |

## 🤖 Modelo final

LightGBM con pérdida `l1` (estima la mediana del precio): 1.500 árboles, `learning_rate=0.015`, `num_leaves=23`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.7`, sin regularización. Rango sugerido con regresión cuantílica (percentiles 7,5 y 92,5).

| Versión | Variables | MdAPE CV (junio) | MdAPE anuncios nuevos sept. (IC 95 %) | Cobertura rango |
|---|---|---|---|---|
| Ridge v1 (línea base) | v1 | – | 25,21 % | – |
| LightGBM v2 | v2 | 20,53 % | 20,14 % (19,02–21,66 %) | 81,5 % |
| LightGBM v2 + ubicación | v2_geo | 19,50 % | 17,74 % (16,87–18,81 %) | 80,5 % |

## 🚀 Instalación y despliegue

**Con Docker (recomendado, p. ej. en una instancia EC2 con Docker y el plugin Compose):**

```bash
git clone <repositorio> && cd precios_airbnb_bogota
docker compose up -d --build
docker compose ps                 # los dos servicios deben estar "running" / "healthy"
# API:     http://<ip>:8001/docs
# Tablero: http://<ip>:8050
```

Abrir los puertos 8001 y 8050 en el grupo de seguridad de la instancia. Para detener: `docker compose down`. Los modelos (`api/modelos/`) y los insumos del tablero (`tablero/datos/`) ya vienen en el repositorio, así que no hace falta descargar datos ni entrenar para desplegar.

**Sin Docker (desarrollo local, Python 3.10 o superior):**

```bash
pip install -e .                          # paquete bogoprice (LightGBM, MLflow, scikit-learn)
pip install -r requirements-modelos.txt   # cuadernos
pip install -r tablero/requirements.txt   # tablero y API
cd api && python -m uvicorn app.main:app --port 8001          # terminal 1 (o ./run.sh)
API_URL=http://localhost:8001 python tablero/app.py          # terminal 2, desde la raíz
```

En PowerShell la variable se define antes: `$env:API_URL="http://localhost:8001"; python tablero/app.py`.

**Datos y MLflow:** copiar `.env.example` como `.env` y poner la URI del servidor MLflow en EC2. Los datos crudos se descargan con `dvc pull` (remoto S3 en `.dvc/config`).

**Reentrenar el modelo final:** ejecutar `notebooks/09_modelo_final.ipynb` (registra en MLflow y regenera `api/modelos/` y `api/requirements.txt`) y luego `python tablero/preparar_tablero.py` para actualizar los insumos del tablero. `python tablero/preparar_tablero.py --reentrenar` hace las dos cosas sin registrar en MLflow.

**Pruebas de la API:** `pip install -r api/test_requirements.txt && python -m pytest`.

---

## 🎯 Objetivo del Proyecto

Construir un modelo predictivo capaz de estimar precios de referencia para alojamientos Airbnb en Bogotá, utilizando variables relacionadas con:

- Ubicación.
- Capacidad del alojamiento.
- Número de habitaciones y camas.
- Tipo de habitación.
- Amenidades disponibles.
- Restricciones de estadía.

La solución permitirá apoyar la toma de decisiones de anfitriones independientes mediante recomendaciones basadas en datos.

---
