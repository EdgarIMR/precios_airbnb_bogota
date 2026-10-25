# 🏢 Predicción de Precios de Referencia para Airbnb en Bogotá

> **Curso:** Despliegue de Soluciones Analíticas (MIAD) - Universidad de los Andes
>
> **Entrega:** Entrega 1 - Análisis Exploratorio y Definición del Proyecto
>
> **Dataset:** Capturas de Inside Airbnb Bogotá (Junio y Septiembre de 2026)

---

## 📌 Descripción del Proyecto

Este proyecto aborda la brecha de información que enfrentan los pequeños anfitriones independientes de Airbnb en Bogotá al definir las tarifas por noche de sus inmuebles.

Para reducir esta desventaja frente a operadores profesionales, se desarrolla una solución analítica de aprendizaje supervisado que estima un **precio de referencia por noche** y un **rango competitivo**, considerando las características físicas, geográficas y de estadía del alojamiento.

La solución final se desplegará mediante una **API de inferencia** y un **tablero interactivo** empaquetados en contenedores **Docker**.

---

## 👥 Integrantes del Equipo (Grupo 6)

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
│
├── .gitignore
├── README.md
│
├── data/
│   ├── listings_junio_2026.csv.gz.dvc
│   └── listings_septiembre_2026.csv.gz.dvc
│
├── notebooks/
│   └── eda.ipynb
│
├── reports/
│   └── figures/
│
└── src/
```

### Descripción de Carpetas

| Carpeta | Descripción |
|----------|-------------|
| `data/` | Datos gestionados mediante DVC |
| `notebooks/` | Análisis exploratorio y experimentación |
| `reports/figures/` | Gráficas utilizadas en el informe |
| `src/` | Código fuente para preprocesamiento y modelado |
| `README.md` | Documentación principal del proyecto |

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
