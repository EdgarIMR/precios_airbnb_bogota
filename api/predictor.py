"""BogoPrice: precio de referencia por noche para alojamientos en Bogotá.

Archivo único y autocontenido para la API y el tablero. No depende del paquete
de entrenamiento (src/bogoprice): solo necesita este archivo, la carpeta
``modelos/`` y las librerías de requirements.txt.

Recibe anuncios con las columnas originales de Inside Airbnb (listings.csv):

    neighbourhood_cleansed, room_type, property_type, accommodates, bedrooms,
    beds, bathrooms_text, amenities, minimum_nights, latitude, longitude

y hace toda la transformación de variables que se usó en el entrenamiento.

Uso:

    from predictor import Predictor

    predictor = Predictor()
    predictor.predecir(anuncio, usar_coordenadas=False)   # un dict -> dict
    predictor.predecir(df_crudo, usar_coordenadas=True)   # un DataFrame -> DataFrame
    predictor.explicar(anuncio, usar_coordenadas=True)    # aporte de cada variable
    predictor.opciones(usar_coordenadas=False)            # listas para el formulario

``usar_coordenadas=True`` usa el modelo con latitud y longitud (más preciso);
``False`` usa el modelo sin ellas (el anfitrión no indica su ubicación exacta).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CARPETA_MODELOS = Path(__file__).resolve().parent / "modelos"

# ==========================================================================================
# 1. TRANSFORMACIÓN DE VARIABLES (igual a la del entrenamiento)
# ==========================================================================================

# Amenidades: columna -> patrón (en minúsculas) sobre cada texto de la lista de Inside Airbnb.
AMENIDADES = {
    # Conjunto v1 (cuaderno 02)
    "tiene_wifi": r"wifi",
    "tiene_parqueadero": r"parking",
    "tiene_ascensor": r"elevator",
    "tiene_zona_trabajo": r"workspace",
    "tiene_agua_caliente": r"^hot water$",
    "tiene_jacuzzi": r"hot tub",
    # Agregadas en v2
    "tiene_cocina": r"^kitchen$",
    "tiene_lavadora": r"washer",
    "tiene_aire_acondicionado": r"air conditioning|^ac ",
    "tiene_piscina": r"pool",
    "tiene_gimnasio": r"gym",
    "tiene_vista": r"view",
    "tiene_balcon": r"patio or balcony",
    "tiene_tv": r"^tv|hdtv",
    "tiene_autocheckin": r"self check-in",
    "acepta_mascotas": r"pets allowed",
}

ETIQUETAS_AMENIDADES = {
    "tiene_wifi": "Wi-Fi", "tiene_parqueadero": "Parqueadero", "tiene_ascensor": "Ascensor",
    "tiene_zona_trabajo": "Zona de trabajo", "tiene_agua_caliente": "Agua caliente",
    "tiene_jacuzzi": "Jacuzzi", "tiene_cocina": "Cocina", "tiene_lavadora": "Lavadora",
    "tiene_aire_acondicionado": "Aire acondicionado", "tiene_piscina": "Piscina",
    "tiene_gimnasio": "Gimnasio", "tiene_vista": "Vista", "tiene_balcon": "Balcón o patio",
    "tiene_tv": "Televisor", "tiene_autocheckin": "Auto check-in",
    "acepta_mascotas": "Acepta mascotas",
}

TIPOS_ALOJAMIENTO = {
    "Entire home/apt": "Alojamiento entero",
    "Private room": "Habitación privada",
    "Hotel room": "Habitación de hotel",
    "Shared room": "Habitación compartida",
}

# Columnas crudas de Inside Airbnb que usa el modelo (las que falten se tratan como vacías)
COLUMNAS_CRUDAS = ["neighbourhood_cleansed", "room_type", "property_type", "accommodates",
                   "bedrooms", "beds", "bathrooms_text", "amenities", "minimum_nights",
                   "latitude", "longitude"]

MAX_NOCHES_MINIMAS = 30          # el formulario ofrece "30 o más" como máximo
UMBRAL_ESTANCIA_MENSUAL = 28     # Airbnb aplica el descuento mensual desde 28 noches
LIMITES_BOGOTA = {"latitude": (4.45, 4.85), "longitude": (-74.25, -73.98)}


def _numero_banos(texto) -> float:
    """'2.5 baths' -> 2.5, 'Shared half-bath' -> 0.5, vacío -> NaN."""
    if pd.isna(texto):
        return np.nan
    texto = str(texto).lower()
    numero = re.search(r"\d+(\.\d+)?", texto)
    if numero:
        return float(numero.group())
    return 0.5 if "half" in texto else np.nan


def _lista_amenidades(valor) -> list[str]:
    """Texto JSON de Inside Airbnb ('["Wifi", "Kitchen"]') o lista de Python -> lista en minúsculas."""
    if isinstance(valor, (list, tuple)):
        return [str(a).lower() for a in valor]
    if pd.isna(valor):
        return []
    valor = str(valor)
    return [a.lower() for a in json.loads(valor)] if valor.startswith("[") else []


def transformar(crudo: pd.DataFrame, tipologias: list[str]) -> pd.DataFrame:
    """Convierte anuncios crudos de Inside Airbnb en las variables del modelo.

    ``tipologias`` es la lista de tipos de inmueble frecuentes en el entrenamiento
    (viene en el paquete del modelo); los demás se agrupan en "Other".
    Si una variable ya viene calculada en la entrada, se respeta."""
    df = crudo.copy()
    for columna in COLUMNAS_CRUDAS:
        if columna not in df:
            df[columna] = np.nan

    # Baños
    if "bathrooms_count" not in df:
        df["bathrooms_count"] = df["bathrooms_text"].map(_numero_banos)
    if "is_shared_bathroom" not in df:
        df["is_shared_bathroom"] = (
            df["bathrooms_text"].fillna("").astype(str).str.lower()
            .str.contains("shared").astype(int))

    # Amenidades
    faltantes = [c for c in AMENIDADES if c not in df]
    if faltantes:
        listas = df["amenities"].map(_lista_amenidades)
        for columna in faltantes:
            patron = re.compile(AMENIDADES[columna])
            df[columna] = listas.map(lambda lst, p=patron: int(any(p.search(a) for a in lst)))

    # Noches mínimas: moda (1) en vacíos, recorte a [1, 30] y marca de estancia mensual
    if "noches_minimas" not in df:
        df["noches_minimas"] = (pd.to_numeric(df["minimum_nights"], errors="coerce")
                                .fillna(1).clip(1, MAX_NOCHES_MINIMAS))
    if "estancia_mensual" not in df:
        df["estancia_mensual"] = (df["noches_minimas"] >= UMBRAL_ESTANCIA_MENSUAL).astype(int)

    # Tipo de inmueble agrupado
    if "tipologia" not in df:
        df["tipologia"] = df["property_type"].where(df["property_type"].isin(tipologias), "Other")

    return df


# ==========================================================================================
# 2. PREDICCIÓN
# ==========================================================================================

# Columnas que se muestran juntas en la explicación
GRUPOS = {
    "latitude": ("ubicacion", "Ubicación exacta"), "longitude": ("ubicacion", "Ubicación exacta"),
    "noches_minimas": ("noches", "Noches mínimas"), "estancia_mensual": ("noches", "Noches mínimas"),
}
ETIQUETAS = {
    "neighbourhood_cleansed": "Localidad", "room_type": "Tipo de alojamiento",
    "tipologia": "Tipo de inmueble", "accommodates": "Huéspedes", "bedrooms": "Habitaciones",
    "beds": "Camas", "bathrooms_count": "Baños", "is_shared_bathroom": "Baño compartido",
    **ETIQUETAS_AMENIDADES,
}


class Predictor:
    """Carga los modelos de ``modelos/`` y predice precio de referencia y rango sugerido."""

    def __init__(self, carpeta: Path | str = CARPETA_MODELOS):
        self.carpeta = Path(carpeta)
        self._paquetes: dict[str, dict] = {}

    def paquete(self, usar_coordenadas: bool = False) -> dict:
        version = "v2_geo" if usar_coordenadas else "v2"
        if version not in self._paquetes:
            ruta = self.carpeta / f"modelo_{version}.joblib"
            if not ruta.exists():
                raise FileNotFoundError(f"No se encontró el modelo {ruta}")
            self._paquetes[version] = joblib.load(ruta)
        return self._paquetes[version]

    # ------------------------------------------------------------------ formulario
    def opciones(self, usar_coordenadas: bool = False) -> dict:
        """Listas para construir el formulario del tablero."""
        p = self.paquete(usar_coordenadas)
        return {
            "localidades": p["localidades"],
            "tipos_alojamiento": TIPOS_ALOJAMIENTO,
            "tipologias": p["tipologias"] + ["Other"],
            "amenidades": ETIQUETAS_AMENIDADES,
            "noches_minimas": {"min": 1, "max": MAX_NOCHES_MINIMAS},
            "huespedes": {"min": 1, "max": 16},
            "limites_coordenadas": LIMITES_BOGOTA if usar_coordenadas else None,
            "metricas_validacion": p["metricas_validacion"],
        }

    # ------------------------------------------------------------------ preparación
    def preparar(self, datos, usar_coordenadas: bool = False) -> tuple[pd.DataFrame, pd.Series]:
        """Transforma los datos crudos y marca qué anuncios se pueden estimar.

        Devuelve (X, motivo): X con las columnas del modelo y ``motivo`` vacío si el
        anuncio es válido, o con la razón por la que no se puede estimar."""
        p = self.paquete(usar_coordenadas)
        crudo = pd.DataFrame([datos]) if isinstance(datos, dict) else pd.DataFrame(datos)
        df = transformar(crudo, p["tipologias"])

        motivo = pd.Series("", index=df.index)
        motivo[~df["neighbourhood_cleansed"].isin(p["localidades"])] = "localidad no habilitada"
        motivo[~df["room_type"].isin(TIPOS_ALOJAMIENTO)] = "tipo de alojamiento no válido"
        motivo[pd.to_numeric(df["accommodates"], errors="coerce").isna()] = "falta accommodates"
        if usar_coordenadas:
            for col, (minimo, maximo) in LIMITES_BOGOTA.items():
                valor = pd.to_numeric(df.get(col), errors="coerce")
                motivo[~valor.between(minimo, maximo)] = f"{col} ausente o fuera de Bogotá"

        X = df.reindex(columns=p["columnas"])
        numericas = [c for c in p["columnas"] if c not in ("neighbourhood_cleansed", "room_type", "tipologia")]
        X[numericas] = X[numericas].apply(pd.to_numeric, errors="coerce")
        return X, motivo

    def predecir(self, datos, usar_coordenadas: bool = False):
        """Precio de referencia por noche (COP) y rango sugerido.

        Con un dict devuelve un dict; con un DataFrame devuelve un DataFrame con una
        fila por anuncio. Los anuncios no válidos quedan sin precio y con su motivo."""
        p = self.paquete(usar_coordenadas)
        X, motivo = self.preparar(datos, usar_coordenadas)
        validos = motivo.eq("")

        salida = pd.DataFrame(index=X.index,
                              columns=["precio_referencia", "rango_inferior", "rango_superior"],
                              dtype=float)
        if validos.any():
            Xv = X[validos]
            precio = np.exp(p["modelo"].predict(Xv))
            inferior = np.minimum(np.exp(p["q_inf"].predict(Xv)), precio)
            superior = np.maximum(np.exp(p["q_sup"].predict(Xv)), precio)
            salida.loc[validos] = np.round(np.column_stack([precio, inferior, superior]), -2)
        salida["motivo_no_estimado"] = motivo
        salida["modelo"] = f"LightGBM {p['version']}"

        if isinstance(datos, dict):
            fila = salida.iloc[0]
            if fila["motivo_no_estimado"]:
                raise ValueError(f"No se puede estimar el anuncio: {fila['motivo_no_estimado']}")
            return {
                "precio_referencia": float(fila["precio_referencia"]),
                "rango_inferior": float(fila["rango_inferior"]),
                "rango_superior": float(fila["rango_superior"]),
                "moneda": "COP",
                "modelo": fila["modelo"],
                "usar_coordenadas": usar_coordenadas,
                "mdape_validacion": _redondear(p["metricas_validacion"].get("mdape_nuevos")),
                "cobertura_rango_validacion": _redondear(p["metricas_validacion"].get("cobertura_nuevos")),
            }
        return salida

    def explicar(self, anuncio: dict, usar_coordenadas: bool = False) -> pd.DataFrame:
        """Aporte de cada variable al precio de un anuncio (gráfico de contribución).

        LightGBM descompone la predicción (log del precio) en un valor base más una
        contribución por columna; se agrupan por variable original y se expresan en
        porcentaje y en pesos."""
        p = self.paquete(usar_coordenadas)
        X, motivo = self.preparar(anuncio, usar_coordenadas)
        if motivo.iloc[0]:
            raise ValueError(f"No se puede estimar el anuncio: {motivo.iloc[0]}")
        prep, modelo = p["modelo"].named_steps["prep"], p["modelo"].named_steps["modelo"]
        contrib = modelo.predict(prep.transform(X), pred_contrib=True)[0]
        precio = float(np.exp(contrib.sum()))

        aportes: dict[str, float] = {}
        etiquetas: dict[str, str] = {}
        for nombre, valor in zip(prep.get_feature_names_out(), contrib[:-1]):
            columna = _columna_original(nombre, p["columnas"])
            clave, etiqueta = GRUPOS.get(columna, (columna, ETIQUETAS.get(columna, columna)))
            aportes[clave] = aportes.get(clave, 0.0) + float(valor)
            etiquetas[clave] = etiqueta

        tabla = pd.DataFrame([{"variable": etiquetas[k], "efecto_pct": (np.exp(c) - 1) * 100,
                               "efecto_cop": precio * (1 - np.exp(-c))} for k, c in aportes.items()])
        tabla = tabla.reindex(tabla["efecto_cop"].abs().sort_values(ascending=False).index)
        tabla.attrs["precio_base_cop"] = float(np.exp(contrib[-1]))
        tabla.attrs["precio_referencia_cop"] = precio
        return tabla.reset_index(drop=True).round({"efecto_pct": 1, "efecto_cop": -2})


def _redondear(valor, decimales: int = 1):
    return None if valor is None else round(float(valor), decimales)


def _columna_original(nombre_transformado: str, columnas: list[str]) -> str:
    """'cat__neighbourhood_cleansed_Chapinero' -> 'neighbourhood_cleansed'."""
    nombre = nombre_transformado.split("__", 1)[-1]
    for col in sorted(columnas, key=len, reverse=True):
        if nombre == col or nombre.startswith(col + "_"):
            return col
    return nombre
