# Imagen del tablero de BogoPrice (se construye desde la carpeta tablero/)
# El tablero consulta el modelo a través de la API (variable API_URL), por lo que no
# necesita LightGBM ni los modelos: solo Dash y los insumos de tablero/datos.
FROM python:3.13-slim

RUN adduser --disabled-password --gecos '' tablero-user

WORKDIR /opt/bogoprice-tablero
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements-docker.txt

COPY app.py servicio.py ./
COPY assets ./assets
COPY datos/anuncios_tablero.csv.gz datos/metricas_modelo.json ./datos/
# Las consultas de la calculadora se guardan en /data (montar un volumen para conservarlas)
RUN mkdir -p /data && chown -R tablero-user:tablero-user ./ /data

ENV API_URL=http://localhost:8001 \
    CONSULTAS_PATH=/data/consultas.csv \
    PORT=8050

USER tablero-user
EXPOSE 8050
CMD ["sh", "-c", "gunicorn app:server --bind 0.0.0.0:${PORT} --workers 2 --timeout 120"]
