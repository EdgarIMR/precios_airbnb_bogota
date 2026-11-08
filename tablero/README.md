# Tablero BogoPrice

Tablero en Dash con dos vistas:

- **Estimar mi precio**: el anfitrión describe su alojamiento (localidad, tipo, capacidad, habitaciones, camas, baños, noches mínimas, baño compartido y amenidades) y recibe el precio de referencia por noche, el rango competitivo (P10–P90) y su ubicación frente a anuncios similares.
- **Mercado por localidad**: precio mediano por localidad, distribución por tipo de alojamiento y mapa de anuncios, con filtros.

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

`servicio.py` aísla el acceso al modelo: hoy carga el `.joblib`; cuando la API esté desplegada (semana 6) basta con definir `API_URL` para que el tablero consulte `POST {API_URL}/predict`.
