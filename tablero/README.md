# Tablero BogoPrice

Tablero en Dash que implementa la maqueta de la Entrega 1 (figuras 5 y 6):

- **Calculadora del anfitrión**: el anfitrión describe su alojamiento (localidad, tipo, huéspedes, habitaciones, camas, baños, baño compartido, noches mínimas y amenidades) y recibe el precio de referencia por noche, el rango sugerido del 80 % (modelos de cuantil P10 y P90), la mediana y el número de comparables, el precio mediano de su tipo en la localidad, un histograma de posición frente a los comparables y la contribución de cada variable en COP.
- **Panel de monitoreo**: estado del modelo (vigente o requiere reentrenamiento según el umbral de MdAPE de 25 %), consultas registradas desde la calculadora, MdAPE fuera de tiempo, cobertura del rango, consultas por localidad y tipo, precio estimado vs. publicado, MdAPE por localidad y evolución del error por captura.

## Diseño

Paleta: azul `#2391FF` para la serie de datos y lo tuyo (tu precio, el ítem activo), con sus tintes pálidos en las áreas; magenta `#F70063` para lo que baja el precio o pasa el umbral; ámbar `#FFCD54` para lo que está entre la meta y el umbral. Tipografía DM Sans, incluida en `assets/fonts` (licencia OFL) para que el tablero funcione sin internet y dentro de Docker.

## Ejecutar

Desde la raíz del repositorio:

```bash
pip install -e .
pip install -r tablero/requirements.txt
python tablero/preparar_tablero.py   # entrena el modelo servible y exporta los datos del tablero
python tablero/app.py                # abre http://localhost:8050
```

## Modelo

`preparar_tablero.py` entrena el mejor modelo lineal registrado en MLflow (`Grupo6_ridge_a0.1`, variables v1 de `config.FEATURES`) y dos modelos de cuantil (P10 y P90) para el rango. Los guarda en `models/modelo.joblib` con `bogoprice.exportar.guardar_modelo`, el mismo formato que leerá la API.

Además exporta `tablero/datos/anuncios_tablero.csv.gz` (anuncios de septiembre con su estimación y rango) y `tablero/datos/metricas_modelo.json` (métricas para el panel de monitoreo). Las consultas se guardan en `tablero/datos/consultas.csv` (fuera de Git; ruta configurable con `CONSULTAS_PATH`).

`servicio.py` aísla el acceso al modelo: hoy carga el `.joblib`; cuando la API esté desplegada (semana 6) basta con definir `API_URL` para que el tablero consulte `POST {API_URL}/predict`.
