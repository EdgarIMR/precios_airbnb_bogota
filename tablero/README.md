# Tablero BogoPrice

Tablero en Dash que implementa la maqueta de la Entrega 1 (figuras 5 y 6) sobre el modelo final del grupo (LightGBM, `api/`).

- **Calculadora de precio**: el anfitrión describe su alojamiento (localidad, tipo de alojamiento e inmueble, huéspedes, habitaciones, camas, baños, noches mínimas y 16 amenidades). Si marca su ubicación exacta en el mapa, se usa el modelo con coordenadas. Recibe el precio de referencia, el rango sugerido (percentiles 7,5 y 92,5), su posición frente a los anuncios comparables y el aporte de cada variable en pesos.
- **Monitoreo del modelo**: estado frente a la meta (MdAPE 20 %) y al umbral de reentrenamiento (25 %), cobertura del rango, error absoluto, error por localidad, consultas registradas desde la calculadora, precio estimado frente al publicado y comparación de versiones (Ridge v1, LightGBM v2 y v2 con ubicación). Las métricas se miden en los anuncios nuevos de septiembre con modelos entrenados solo con junio.

## Ejecutar

Desde la raíz del repositorio:

```bash
pip install -e .
pip install -r tablero/requirements.txt
python tablero/preparar_tablero.py          # métricas del panel; entrena api/modelos si faltan

# Opción A: tablero consultando la API (dos terminales)
cd api && ./run.sh                          # terminal 1: API en http://localhost:8001
API_URL=http://localhost:8001 python tablero/app.py   # terminal 2: tablero en http://localhost:8050

# Opción B: tablero solo, con el mismo predictor cargado localmente
python tablero/app.py
```

`servicio.py` decide de dónde sale el precio: con `API_URL` llama a `POST /api/v1/predict` y `/api/v1/explicar`; sin ella importa `api/predictor.py`. El pie de página del tablero dice cuál se está usando. Las consultas se guardan en `tablero/datos/consultas.csv` (fuera de Git; ruta configurable con `CONSULTAS_PATH`). El botón de MLflow usa `MLFLOW_UI_URL`.

## Diseño

Paleta: azul `#2391FF` para la serie de datos y lo tuyo (tu precio, el ítem activo), con sus tintes pálidos en las áreas; magenta `#F70063` para lo que baja el precio o pasa el umbral; ámbar `#FFCD54` para lo que está entre la meta y el umbral. Tipografía DM Sans, incluida en `assets/fonts` (licencia OFL) para que el tablero funcione sin internet y dentro de Docker.
