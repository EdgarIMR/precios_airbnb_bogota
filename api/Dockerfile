# Imagen de la API de BogoPrice (se construye desde la carpeta api/)
FROM python:3.13-slim

# LightGBM necesita la librería OpenMP en tiempo de ejecución
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Usuario sin privilegios que ejecuta la app
RUN adduser --disabled-password --gecos '' api-user

WORKDIR /opt/bogoprice-api
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY predictor.py run.sh ./
COPY app ./app
COPY modelos ./modelos
RUN chmod +x run.sh && chown -R api-user:api-user ./

USER api-user
EXPOSE 8001
CMD ["bash", "./run.sh"]
