# api/ — modelo de BogoPrice listo para el tablero

Carpeta autocontenida: **no depende del resto del repositorio**. Se puede copiar sola a un contenedor Docker o a otra máquina.

```
api/
├── predictor.py          ← transformación de variables + predicción (archivo único)
├── requirements.txt      ← versiones exactas con las que se entrenaron los modelos
└── modelos/
    ├── modelo_v2.joblib       ← sin coordenadas
    └── modelo_v2_geo.joblib   ← con latitud y longitud
```

## Instalación

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
# {'precio_referencia': 149500.0, 'rango_inferior': 130200.0, 'rango_superior': 220900.0,
#  'moneda': 'COP', 'modelo': 'LightGBM v2_geo', 'usar_coordenadas': True,
#  'mdape_validacion': 17.4, 'cobertura_rango_validacion': 80.5}
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

LightGBM con pérdida `l1` (optimiza la mediana del error), entrenado con las capturas de junio y septiembre de 2026. Validación (entrenado con junio, evaluado con anuncios nuevos de septiembre):

| `usar_coordenadas` | MdAPE | Cobertura del rango sugerido |
|---|---|---|
| `False` | 19,3 % | 81,5 % |
| `True` | 17,4 % | 80,5 % |

Para reentrenar, ejecutar `notebooks/09_modelo_final.ipynb` desde la raíz del repositorio: regenera los dos `.joblib` y `requirements.txt`, y verifica que la transformación de `predictor.py` sea idéntica a la del entrenamiento.
