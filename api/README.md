# api/: modelo de BogoPrice y su API

Carpeta autocontenida: **no depende del resto del repositorio**. Se puede copiar sola a un contenedor Docker o a otra máquina.

```
api/
├── predictor.py          ← transformación de variables + predicción (archivo único)
├── app/                  ← servidor FastAPI (main.py, api.py, schemas/, tests/)
├── run.sh · Dockerfile   ← arranque en el puerto 8001 y contenedor
├── requirements.txt      ← versiones exactas con las que se entrenaron los modelos + FastAPI
└── modelos/
    ├── modelo_v2.joblib       ← sin coordenadas
    └── modelo_v2_geo.joblib   ← con latitud y longitud
```

## Servidor de la API (FastAPI)

```bash
cd api
pip install -r requirements.txt
./run.sh                      # http://localhost:8001, documentación en /docs
# En Windows: python -m uvicorn app.main:app --port 8001
```

Rutas (prefijo `/api/v1`):

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/health` | Nombre, versión de la API y modelos cargados |
| GET | `/opciones?usar_coordenadas=false` | Localidades, tipos, tipologías y amenidades para el formulario, y métricas de validación |
| POST | `/predict` | Precio de referencia y rango sugerido. Cuerpo: columnas de Inside Airbnb y `usar_coordenadas` |
| POST | `/explicar` | Aporte de cada variable al precio, en % y en pesos |

Un anuncio que no se puede estimar (localidad no habilitada, coordenadas fuera de Bogotá) responde 422 con el motivo.

Pruebas: `pip install -r test_requirements.txt && python -m pytest app/tests`.

Contenedor:

```bash
docker build -t bogoprice-api api/
docker run -d -p 8001:8001 --name bogoprice-api bogoprice-api
```

Para levantar API y tablero juntos, usar `docker compose up -d --build` desde la raíz del repositorio (ver el README principal).

## Uso del predictor desde Python

### Instalación

```bash
pip install -r api/requirements.txt
```

## Uso

```python
import sys
sys.path.insert(0, "api")          # o ejecutar desde dentro de api/
from predictor import Predictor

predictor = Predictor()
```

### Entrada: datos crudos de Inside Airbnb

`predictor.py` recibe las columnas **tal como vienen en `listings.csv`** de Inside Airbnb y hace toda la transformación de variables (baños, amenidades, noches mínimas, estancia mensual y tipología):

| Columna | Ejemplo | ¿Obligatoria? |
|---|---|---|
| `neighbourhood_cleansed` | `"Chapinero"` | Sí (una de las localidades habilitadas) |
| `room_type` | `"Entire home/apt"` | Sí |
| `accommodates` | `2` | Sí |
| `property_type` | `"Entire rental unit"` | No (si falta o es poco común → `"Other"`) |
| `bedrooms`, `beds` | `1`, `1` | No |
| `bathrooms_text` | `"1 bath"`, `"1.5 shared baths"` | No |
| `amenities` | `'["Wifi", "Kitchen"]'` o `["Wifi", "Kitchen"]` | No |
| `minimum_nights` | `2` | No (si falta → 1) |
| `latitude`, `longitude` | `4.6486`, `-74.0603` | Solo con `usar_coordenadas=True` |

### Un anuncio (dict) → dict

```python
anuncio = {"neighbourhood_cleansed": "Chapinero", "room_type": "Entire home/apt",
           "property_type": "Entire rental unit", "accommodates": 2, "bedrooms": 1,
           "beds": 1, "bathrooms_text": "1 bath",
           "amenities": ["Wifi", "Kitchen", "Elevator", "Dedicated workspace"],
           "minimum_nights": 2, "latitude": 4.6486, "longitude": -74.0603}

predictor.predecir(anuncio, usar_coordenadas=True)
# {'precio_referencia': ..., 'rango_inferior': ..., 'rango_superior': ...,
#  'moneda': 'COP', 'modelo': 'LightGBM v2_geo', 'usar_coordenadas': True,
#  'mdape_validacion': 17.7, 'cobertura_rango_validacion': 80.5}
```

Si el anuncio no se puede estimar (localidad no habilitada, faltan coordenadas con `usar_coordenadas=True`, etc.), lanza `ValueError` con el motivo.

### Varios anuncios (DataFrame del CSV crudo) → DataFrame

```python
import pandas as pd
crudo = pd.read_csv("listings.csv.gz")
predictor.predecir(crudo, usar_coordenadas=False)
```

Devuelve una fila por anuncio con `precio_referencia`, `rango_inferior`, `rango_superior` y `motivo_no_estimado` (vacío si se pudo estimar).

### Otras funciones

- `predictor.explicar(anuncio, usar_coordenadas)` → aporte de cada variable al precio, en % y en pesos (gráfico de contribución del tablero).
- `predictor.opciones(usar_coordenadas)` → localidades habilitadas, tipos de alojamiento, tipologías y amenidades para el formulario, y las métricas de validación del modelo.

## Modelos

LightGBM con pérdida `l1` (optimiza la mediana del error), 1.500 árboles, `learning_rate=0.015`, `num_leaves=23`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.7` (elegidos por validación cruzada de 10 pliegues en el cuaderno 04), entrenado con las capturas de junio y septiembre de 2026. Prueba final (entrenado con junio, evaluado con los 1.632 anuncios nuevos de septiembre):

| `usar_coordenadas` | MdAPE (IC 95 %) | MAE | Cobertura del rango sugerido |
|---|---|---|---|
| `False` | 20,1 % (19,0–21,7 %) | $53.853 | 81,5 % |
| `True` | 17,7 % (16,9–18,8 %) | $50.467 | 80,5 % |

Para reentrenar, ejecutar `notebooks/09_modelo_final.ipynb` desde la raíz del repositorio (o `python tablero/preparar_tablero.py --reentrenar`): regenera los dos `.joblib` y `requirements.txt`, y verifica que la transformación de `predictor.py` sea idéntica a la del entrenamiento.
