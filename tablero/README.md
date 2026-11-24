# Tablero BogoPrice

Tablero en Dash que implementa la maqueta de la Entrega 1 (figuras 5 y 6) sobre el modelo final del grupo (LightGBM `l1`, 23 hojas, `learning_rate=0.015`, servido por la API de `api/`).

- **Calculadora de precio**: el anfitrión describe su alojamiento (localidad, tipo de alojamiento e inmueble, huéspedes, habitaciones, camas, baños, noches mínimas y 16 amenidades). Si marca su ubicación exacta en el mapa, se usa el modelo con coordenadas. Recibe el precio de referencia, el rango sugerido (percentiles 7,5 y 92,5), su posición frente a los anuncios comparables y el aporte de cada variable en pesos.
- **Monitoreo del modelo**: estado frente a la meta (MdAPE 20 %) y al umbral de reentrenamiento (25 %), cobertura del rango, error absoluto, error por localidad, consultas registradas desde la calculadora, precio estimado frente al publicado y comparación con los otros modelos (mediana por localidad y tipo, Ridge, LightGBM sin y con ubicación) y un recuadro que explica cómo leer las métricas. Las métricas se miden en los 1.632 anuncios nuevos de septiembre con modelos entrenados solo con junio (v2: 20,14 %; v2 + ubicación: 17,74 %).

## Ejecutar con Docker (recomendado)

Desde la raíz del repositorio, con Docker y Docker Compose instalados:

```bash
docker compose up -d --build     # API en :8001 y tablero en :8050
```

El tablero se conecta a la API por `API_URL=http://api:8001` y espera a que esté sana. Las consultas se guardan en el volumen `consultas`. Variable opcional: `MLFLOW_UI_URL` (enlace del botón "Ver experimentos en MLflow").

Solo el tablero, apuntando a una API ya desplegada:

```bash
docker build -t bogoprice-tablero tablero/
docker run -d -p 8050:8050 -e API_URL=http://<ip-api>:8001 --name bogoprice-tablero bogoprice-tablero
```

## Ejecutar sin Docker

Desde la raíz del repositorio:

```bash
pip install -e .
pip install -r tablero/requirements.txt

# Opción A: tablero consultando la API (dos terminales)
cd api && ./run.sh                                       # terminal 1: API en http://localhost:8001
API_URL=http://localhost:8001 python tablero/app.py      # terminal 2: tablero en http://localhost:8050

# Opción B: tablero solo, con el mismo predictor cargado localmente
python tablero/app.py
```

`servicio.py` decide de dónde sale el precio: con `API_URL` llama a `GET /api/v1/opciones`, `POST /api/v1/predict` y `/api/v1/explicar` (reintenta mientras la API arranca); sin ella importa `api/predictor.py`. Al arrancar, la consola indica cuál se está usando. Las consultas se guardan en `tablero/datos/consultas.csv` (fuera de Git; ruta configurable con `CONSULTAS_PATH`). El puerto se cambia con `PORT`.

## Regenerar los insumos

`tablero/datos/anuncios_tablero.csv.gz` y `metricas_modelo.json` ya vienen en el repositorio. Para regenerarlos (y reentrenar los modelos de `api/modelos`):

```bash
python tablero/preparar_tablero.py --reentrenar
```

## Diseño

Paleta: azul `#2391FF` para la serie de datos y lo tuyo (tu precio, el ítem activo), con sus tintes pálidos en las áreas; magenta `#F70063` para lo que baja el precio o pasa el umbral; ámbar `#FFCD54` para lo que está entre la meta y el umbral. Tipografía DM Sans, incluida en `assets/fonts` (licencia OFL) para que el tablero funcione sin internet y dentro de Docker.
