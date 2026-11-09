"""Esquemas de entrada y salida. La entrada usa las columnas de listings.csv de Inside Airbnb."""
from __future__ import annotations

from pydantic import BaseModel, Field

EJEMPLO = {
    "neighbourhood_cleansed": "Chapinero",
    "room_type": "Entire home/apt",
    "property_type": "Entire rental unit",
    "accommodates": 2,
    "bedrooms": 1,
    "beds": 1,
    "bathrooms_text": "1 bath",
    "amenities": ["Wifi", "Kitchen", "Elevator", "Dedicated workspace", "Hot water", "TV"],
    "minimum_nights": 2,
    "latitude": 4.6486,
    "longitude": -74.0603,
    "usar_coordenadas": False,
}


class Anuncio(BaseModel):
    neighbourhood_cleansed: str = Field(description="Localidad habilitada (ver /opciones)")
    room_type: str = Field(description="Entire home/apt, Private room, Hotel room o Shared room")
    accommodates: int = Field(ge=1, le=16, description="Número de huéspedes")
    property_type: str | None = Field(None, description="Tipo de inmueble de Inside Airbnb; si es poco común se agrupa en Other")
    bedrooms: float | None = Field(None, ge=0)
    beds: float | None = Field(None, ge=0)
    bathrooms_text: str | None = Field(None, description='Texto de Inside Airbnb, p. ej. "1.5 shared baths"')
    amenities: list[str] = Field(default_factory=list, description="Amenidades con los nombres de Inside Airbnb")
    minimum_nights: int | None = Field(None, ge=1)
    latitude: float | None = None
    longitude: float | None = None
    usar_coordenadas: bool = Field(False, description="True usa el modelo con latitud y longitud")

    model_config = {"json_schema_extra": {"examples": [EJEMPLO]}}

    def crudo(self) -> dict:
        datos = self.model_dump(exclude={"usar_coordenadas"})
        return {k: v for k, v in datos.items() if v is not None}


class Prediccion(BaseModel):
    precio_referencia: float
    rango_inferior: float
    rango_superior: float
    moneda: str
    modelo: str
    usar_coordenadas: bool
    mdape_validacion: float | None
    cobertura_rango_validacion: float | None


class Aporte(BaseModel):
    variable: str
    efecto_pct: float
    efecto_cop: float


class Explicacion(BaseModel):
    precio_base_cop: float
    precio_referencia_cop: float
    aportes: list[Aporte]


class Opciones(BaseModel):
    localidades: list[str]
    tipos_alojamiento: dict[str, str]
    tipologias: list[str]
    amenidades: dict[str, str]
    noches_minimas: dict[str, int]
    huespedes: dict[str, int]
    limites_coordenadas: dict[str, list[float]] | None
    metricas_validacion: dict[str, float]
