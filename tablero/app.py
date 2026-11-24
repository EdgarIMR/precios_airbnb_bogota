"""Tablero BogoPrice (maqueta de la Entrega 1, figuras 5 y 6).

Dos pantallas:
  /             Calculadora: el anfitrión describe su alojamiento y recibe el precio
                de referencia, el rango donde se publican 8 de cada 10 alojamientos
                parecidos, su posición frente a los comparables y qué variables mueven
                el precio.
  /monitoreo    Monitoreo del modelo: error fuera de tiempo frente a la meta y al
                límite aceptable, rango sugerido, error por localidad (con o sin ubicación),
                consultas registradas y precio estimado frente al publicado.

Ejecutar desde la raíz del repo:
    python tablero/preparar_tablero.py   # una vez: modelo + insumos
    python tablero/app.py                # http://localhost:8050
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from servicio import contribuciones, opciones, origen, predecir

AQUI = Path(__file__).resolve().parent
DATOS = pd.read_csv(AQUI / "datos" / "anuncios_tablero.csv.gz")
METRICAS = json.loads((AQUI / "datos" / "metricas_modelo.json").read_text(encoding="utf-8"))
CONSULTAS = Path(os.getenv("CONSULTAS_PATH", AQUI / "datos" / "consultas.csv"))
MLFLOW_UI = os.getenv("MLFLOW_UI_URL", "http://100.53.0.40:8050")
OPCIONES = opciones()
print(f"Tablero BogoPrice: predicciones desde {origen()}", flush=True)
# Antes venían de bogoprice.config; se leen de los insumos para que el tablero funcione solo
# (por ejemplo dentro de su contenedor, sin el paquete de entrenamiento).
PRECIO = "price"
MIN_ANUNCIOS_LOCALIDAD = METRICAS.get("min_anuncios_localidad", 500)

TIPOS = {
    "Entire home/apt": "Alojamiento completo",
    "Private room": "Habitación privada",
    "Hotel room": "Habitación de hotel",
    "Shared room": "Habitación compartida",
}
PLURAL = {
    "Entire home/apt": "alojamientos completos",
    "Private room": "habitaciones privadas",
    "Hotel room": "habitaciones de hotel",
    "Shared room": "habitaciones compartidas",
}
# Prefijo de property_type en Inside Airbnb para cada tipo de alojamiento
PREFIJO_TIPO = {"Entire home/apt": "Entire", "Private room": "Private room",
                "Hotel room": "Room in", "Shared room": "Shared room"}
TIPOLOGIAS = {
    "Entire rental unit": "Apartamento", "Entire condo": "Apartamento en conjunto",
    "Entire home": "Casa", "Entire loft": "Loft", "Entire serviced apartment": "Apartamento con servicios",
    "Entire guesthouse": "Casa de huéspedes", "Private room in rental unit": "En un apartamento",
    "Private room in condo": "En un apartamento de conjunto", "Private room in home": "En una casa",
    "Private room in loft": "En un loft", "Private room in serviced apartment": "En un apartamento con servicios",
    "Private room in guesthouse": "En una casa de huéspedes", "Private room in bed and breakfast": "En un bed and breakfast",
    "Private room in casa particular": "En una casa particular", "Room in hotel": "Hotel",
    "Room in boutique hotel": "Hotel boutique", "Shared room in hotel": "En un hotel", "Other": "Otro",
}
LOCALIDADES = OPCIONES["localidades"]
NOMBRE_LOC = {"Engativa": "Engativá", "Fontibon": "Fontibón", "Usaquen": "Usaquén"}
AMENIDADES = OPCIONES["amenidades"]
# Nombre de cada amenidad tal como aparece en Inside Airbnb (lo que espera la API)
NOMBRE_AIRBNB = {
    "tiene_wifi": "Wifi", "tiene_parqueadero": "Free parking on premises", "tiene_ascensor": "Elevator",
    "tiene_zona_trabajo": "Dedicated workspace", "tiene_agua_caliente": "Hot water", "tiene_jacuzzi": "Hot tub",
    "tiene_cocina": "Kitchen", "tiene_lavadora": "Washer", "tiene_aire_acondicionado": "Air conditioning",
    "tiene_piscina": "Pool", "tiene_gimnasio": "Gym", "tiene_vista": "City skyline view",
    "tiene_balcon": "Patio or balcony", "tiene_tv": "TV", "tiene_autocheckin": "Self check-in",
    "acepta_mascotas": "Pets allowed",
}
CENTROS = DATOS.groupby("neighbourhood_cleansed")[["latitude", "longitude"]].median()
LIMITES = {"latitude": (4.45, 4.85), "longitude": (-74.25, -73.98)}
NOCHES = [1, 2, 3, 4, 7, 14, 30]
AZUL, AZUL_200, MAGENTA, TINTA, GRIS, REGLA = (
    "#2391ff", "#b8d8ff", "#f70063", "#0f1b2d", "#566376", "#edf2f8")
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
META, UMBRAL = METRICAS["meta_mdape"], METRICAS["umbral_reentrenamiento"]
EV, EVG = METRICAS["evaluacion"]["v2"], METRICAS["evaluacion"]["v2_geo"]
FINAL = METRICAS["entrenamiento_final"]
P_INF, P_SUP = (p * 100 for p in METRICAS["percentiles_rango"])


def loc(nombre: str) -> str:
    return NOMBRE_LOC.get(nombre, nombre)


def cop(valor: float) -> str:
    return f"${valor:,.0f}".replace(",", ".")


def num(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", ".")


def pct(valor: float, dec: int = 1) -> str:
    return f"{valor:.{dec}f} %".replace(".", ",")


def fecha(iso: str) -> str:
    a, m, d = iso.split("-")
    return f"{int(d)} {MESES[int(m) - 1]} {a}"


def svg(cuerpo: str, color: str = TINTA, tam: int = 16) -> str:
    """Ícono de trazo 1,6 px como data URI (Dash no tiene componentes SVG)."""
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{tam}" height="{tam}" viewBox="0 0 24 24" '
         f'fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
         f'{cuerpo}</svg>')
    return "data:image/svg+xml;utf8," + quote(s)


# Marca: un techo con el punto de una etiqueta de precio, en el azul de la paleta
MARCA = "data:image/svg+xml;utf8," + quote(
    '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28">'
    '<rect width="28" height="28" rx="7" fill="#2391ff"/>'
    '<path d="M6 14.5 14 7l8 7.5V21a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1z" fill="#fff"/>'
    '<circle cx="14" cy="16" r="2.2" fill="#2391ff"/></svg>')
ICONO_CALC = '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M8 7h8M8 11h2M14 11h2M8 15h2M14 15h2M8 18h2M14 18h2"/>'
ICONO_MON = '<path d="M3 12h4l2-6 4 12 2-6h6"/>'


# --- Armazón ---------------------------------------------------------------------------
def lateral(ruta: str):
    def item(href, texto, icono):
        activo = ruta == href
        return dcc.Link([html.Img(src=svg(icono, "#0a4f99" if activo else GRIS), alt=""), texto],
                        href=href, className="nav activo" if activo else "nav")
    return html.Nav([
        html.Div([html.Img(src=MARCA, alt=""), html.Span("BogoPrice", className="marca-nombre")],
                 className="marca"),
        html.Div([html.Strong("Bogotá"),
                  f"{len(LOCALIDADES)} localidades disponibles"],
                 className="ambito"),
        html.Div("Anfitriones", className="grupo"),
        item("/", "Calculadora de precio", ICONO_CALC),
        html.Div("Equipo del proyecto", className="grupo"),
        item("/monitoreo", "Monitoreo del modelo", ICONO_MON),
        html.Div([html.Strong("Fuente de datos"),
                  f"Inside Airbnb, Bogotá. {num(FINAL['n_anuncios'])} anuncios publicados en junio y "
                  "septiembre de 2026."],
                 className="lateral-pie"),
    ], className="lateral", **{"aria-label": "Navegación"})


def barra_movil(ruta: str):
    otra = ("/monitoreo", "Monitoreo") if ruta == "/" else ("/", "Calculadora")
    return html.Div([
        html.Div([html.Img(src=MARCA, alt="", style={"width": "22px", "height": "22px"}),
                  html.Span("BogoPrice", className="marca-nombre")], className="marca"),
        dcc.Link(otra[1], href=otra[0]),
    ], className="barra-movil")


def campo(etiqueta, control, ayuda=None, para=None):
    hijos = [html.Label(etiqueta, className="etiqueta", htmlFor=para), control]
    if ayuda:
        hijos.append(html.Div(ayuda, className="ayuda"))
    return html.Div(hijos, className="campo")


def numero(id_, valor, minimo, maximo, paso=1):
    return dcc.Input(id=id_, type="number", value=valor, min=minimo, max=maximo, step=paso,
                     className="entrada")


def fig_base(alto=260):
    f = go.Figure()
    f.update_layout(template="none", height=alto, margin=dict(t=8, l=8, r=8, b=8), showlegend=False,
                    font=dict(family="DM Sans, system-ui, sans-serif", size=12, color=GRIS),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", separators=",.")
    f.update_xaxes(showgrid=False, linecolor="#cdd8e5", ticks="outside", tickcolor="#cdd8e5", zeroline=False)
    f.update_yaxes(showgrid=True, gridcolor=REGLA, zeroline=False, linecolor="rgba(0,0,0,0)")
    return f


TIPOLOGIA_COMUN = {"Entire home/apt": "Entire rental unit", "Private room": "Private room in rental unit",
                   "Hotel room": "Room in hotel", "Shared room": "Shared room in hotel"}


def opciones_tipologia(tipo):
    prefijo = PREFIJO_TIPO[tipo]
    return [{"label": TIPOLOGIAS.get(t, t), "value": t}
            for t in OPCIONES["tipologias"] if t.startswith(prefijo)] + [{"label": "Otro", "value": "Other"}]


# Rejilla invisible sobre Bogotá: en un mapa de Plotly solo se puede hacer clic sobre puntos
_lat = np.arange(LIMITES["latitude"][0], LIMITES["latitude"][1], 0.004)
_lon = np.arange(LIMITES["longitude"][0], LIMITES["longitude"][1], 0.004)
REJILLA = np.array(np.meshgrid(_lat, _lon)).reshape(2, -1)


def fig_mapa(lat, lon):
    f = go.Figure()
    f.add_trace(go.Scattermap(lat=REJILLA[0], lon=REJILLA[1], mode="markers",
                              marker=dict(size=14, opacity=0.01, color=AZUL), hoverinfo="none"))
    f.add_trace(go.Scattermap(lat=[lat], lon=[lon], mode="markers",
                              marker=dict(size=14, color=AZUL), hoverinfo="none"))
    f.update_layout(map=dict(style="carto-positron", center=dict(lat=lat, lon=lon), zoom=12.5),
                    margin=dict(t=0, l=0, r=0, b=0), height=220, showlegend=False, clickmode="event")
    return f


# --- Pantalla 1: calculadora ----------------------------------------------------------------
def pantalla_calculadora():
    centro = CENTROS.loc["Chapinero"]
    formulario = html.Div([
        html.H2("Tu alojamiento"),
        html.P("Completa los datos y pulsa «Calcular precio».", className="ayuda",
               style={"marginTop": "-4px", "marginBottom": "12px"}),
        campo("Localidad", dcc.Dropdown([{"label": loc(l), "value": l} for l in LOCALIDADES], "Chapinero",
                                        id="f-localidad", clearable=False),
              f"Se incluyen las localidades con al menos {MIN_ANUNCIOS_LOCALIDAD} anuncios. En las demás "
              "hay muy pocos datos para dar un precio confiable."),
        dcc.Checklist([{"label": "Marcar la ubicación exacta en el mapa", "value": 1}], [], id="f-geo",
                      className="casillas"),
        html.Div([
            dcc.Graph(id="f-mapa", figure=fig_mapa(centro.latitude, centro.longitude),
                      config={"displayModeBar": False, "scrollZoom": True}),
            html.Div("Haz clic en el mapa donde está tu alojamiento. Con la ubicación exacta el precio "
                     "estimado es más preciso.", className="ayuda"),
        ], id="f-mapa-caja", className="mapa-caja", style={"display": "none"}),
        dcc.Store(id="f-coords", data={"lat": centro.latitude, "lon": centro.longitude}),
        campo("Tipo de alojamiento", dcc.Dropdown([{"label": v, "value": k} for k, v in TIPOS.items()],
                                                  "Entire home/apt", id="f-tipo", clearable=False)),
        campo("Tipo de inmueble", dcc.Dropdown(opciones_tipologia("Entire home/apt"), "Entire rental unit",
                                               id="f-tipologia", clearable=False),
              "Si tu inmueble no aparece en la lista, elige «Otro»."),
        html.Div([
            campo("Huéspedes", numero("f-huespedes", 2, 1, 16), para="f-huespedes"),
            campo("Habitaciones", numero("f-habitaciones", 1, 0, 10), para="f-habitaciones"),
            campo("Camas", numero("f-camas", 1, 0, 16), para="f-camas"),
            campo("Baños", numero("f-banos", 1, 0, 8, 0.5), para="f-banos"),
        ], className="par"),
        campo("Noches mínimas",
              dcc.RadioItems([{"label": "30+" if n == 30 else str(n), "value": n} for n in NOCHES], 2,
                             id="f-noches", className="botonera"),
              "Con 28 noches o más Airbnb aplica el descuento mensual, por eso el precio por noche "
              "suele ser menor."),
        campo("Baño", dcc.Checklist([{"label": "Es compartido", "value": 1}], [], id="f-compartido",
                                    className="casillas")),
        campo("Amenidades", dcc.Checklist([{"label": v, "value": k} for k, v in AMENIDADES.items()],
                                          ["tiene_wifi", "tiene_agua_caliente", "tiene_zona_trabajo",
                                           "tiene_ascensor", "tiene_cocina", "tiene_tv"],
                                          id="f-amenidades", className="casillas dos-col")),
        html.Button("Calcular precio", id="f-boton", n_clicks=0, className="boton"),
    ], className="tarjeta formulario")

    lider = html.Div([
        html.Div(id="r-etq", className="lider-etq"),
        html.Div([html.Span(id="r-precio"), html.Small(" COP")], className="precio cifra"),
        html.P(id="r-rango", className="rango-texto"),
        html.P("Es una referencia basada en lo que cobran alojamientos parecidos; no garantiza reservas. "
               "No tiene en cuenta temporadas ni fechas especiales.", className="aviso",
               style={"marginTop": "4px"}),
        html.Div(id="r-escala", className="escala", role="img",
                 **{"aria-label": "Rango sugerido frente a alojamientos parecidos"}),
        html.Div([
            html.Div([html.Div("Precio típico de alojamientos parecidos", className="kpi-etq"),
                      html.Div(id="r-mediana", className="kpi-valor"),
                      html.Div(id="r-mediana-det", className="kpi-det")]),
            html.Div([html.Div("Alojamientos parecidos", className="kpi-etq"),
                      html.Div(id="r-n", className="kpi-valor"),
                      html.Div(id="r-n-det", className="kpi-det")]),
            html.Div([html.Div("Precio típico en la localidad", className="kpi-etq"),
                      html.Div(id="r-loc", className="kpi-valor"),
                      html.Div(id="r-loc-det", className="kpi-det")]),
        ], className="franja"),
    ], className="tarjeta lider")

    paneles = html.Div([
        html.Div([
            html.Div([html.H2("Dónde queda tu precio"),
                      html.P("Cuántos alojamientos parecidos cobran cada precio por noche. La franja azul es "
                             "tu rango sugerido y la línea, tu precio.")],
                     className="tarjeta-cab"),
            html.Div(dcc.Graph(id="r-histograma", config={"displayModeBar": False}), className="tarjeta-cuerpo"),
        ], className="tarjeta"),
        html.Div([
            html.Div([html.H2("Qué mueve tu precio"),
                      html.P("Cuánto suma o resta cada característica de tu alojamiento, en pesos por noche, "
                             "frente al precio promedio. Son valores aproximados: no garantizan que el precio "
                             "cambie exactamente en esa cantidad.")], className="tarjeta-cab"),
            html.Div(id="r-contribuciones", className="barras"),
        ], className="tarjeta"),
    ], className="dos")

    titulo = html.Div([
        html.Div([html.H1("Calculadora de precio"), html.P(id="r-hallazgo", className="hallazgo")]),
        html.Span(f"Datos al {fecha(METRICAS['fecha_captura_evaluacion'])}", className="chip chip-neutro"),
    ], className="titulo")

    return html.Div([
        titulo,
        html.Div([formulario, dcc.Loading(
            html.Div([lider, html.Div(id="r-error"), paneles], className="resultado"),
            delay_show=150,
            overlay_style={"visibility": "visible", "opacity": 0.35, "filter": "blur(1px)"},
            custom_spinner=html.Div([html.Div(className="girador"), html.Span("Calculando precio…")],
                                    className="calculando"),
            parent_style={"minWidth": 0},
        )],
                 className="calc"),
    ])


def escala(r, precios):
    """Rango sugerido y tu precio sobre la escala P3-P97 de los comparables."""
    lo, hi = float(precios.quantile(0.03)), float(precios.quantile(0.97))
    lo, hi = min(lo, r["inferior"], r["precio"]), max(hi, r["superior"], r["precio"])
    pos = lambda v: f"{(v - lo) / (hi - lo) * 100:.2f}%"
    med = float(precios.median())
    return [
        html.Div(className="escala-pista"),
        html.Div(className="escala-rango", style={"left": pos(r["inferior"]),
                                                  "width": f"{(r['superior'] - r['inferior']) / (hi - lo) * 100:.2f}%"}),
        html.Div(className="escala-marca", style={"left": pos(med)}),
        html.Div(f"Precio típico {cop(med)}", className="escala-etq abajo", style={"left": pos(med)}),
        html.Div(className="escala-tu", style={"left": pos(r["precio"])}),
        html.Div(f"Tu precio {cop(r['precio'])}", className="escala-etq arriba tu", style={"left": pos(r["precio"])}),
        html.Div(cop(lo), className="escala-borde", style={"left": "0"}),
        html.Div(cop(hi), className="escala-borde", style={"right": "0"}),
    ]


def lista_contribuciones(c: pd.DataFrame):
    c = c.reindex(c["cop"].abs().sort_values(ascending=False).index).head(7)
    tope = c["cop"].abs().max() or 1
    filas = []
    for _, f in c.iterrows():
        sube = f["cop"] >= 0
        barra = html.Div(className="barra" + ("" if sube else " neg"),
                         style={"width": f"{abs(f['cop']) / tope * 88:.1f}%"})
        filas.append(html.Div([
            html.Div([html.Span(f["variable"]),
                      html.Span(("+" if sube else "−") + cop(abs(f["cop"])), className="barra-valor",
                                style={"color": "#0a4f99" if sube else "#b8004a"})], className="barra-cab"),
            html.Div([html.Div(None if sube else barra, className="izq"),
                      html.Div(barra if sube else None, className="der")], className="diverge"),
        ], className="barra-fila"))
    return filas


def registrar_consulta(anuncio, r):
    nuevo = not CONSULTAS.exists()
    CONSULTAS.parent.mkdir(parents=True, exist_ok=True)
    with CONSULTAS.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha", "localidad", "tipo", "huespedes", "precio_estimado", "modelo"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), anuncio["neighbourhood_cleansed"],
                    anuncio["room_type"], anuncio["accommodates"], round(r["precio"]), r["modelo"]])


# --- Pantalla 2: monitoreo --------------------------------------------------------------------
MODELOS = {"v2_geo": "Con ubicación", "v2": "Sin ubicación"}
POR_LOC = METRICAS["mdape_por_localidad"]


def selector_modelo(id_):
    """Botones para elegir qué modelo se grafica (con o sin ubicación)."""
    return html.Div([
        html.Span("Modelo", className="selector-etq"),
        dcc.RadioItems([{"label": v, "value": k} for k, v in MODELOS.items()], "v2_geo",
                       id=id_, className="botonera"),
    ], className="selector")


def pantalla_monitoreo():
    vigente = EV["mdape_nuevos"] <= UMBRAL

    titulo = html.Div([
        html.Div([html.H1("Monitoreo del modelo"),
                  html.Span("Modelo al día" if vigente else "Necesita actualizarse",
                            className="chip " + ("chip-ok" if vigente else "chip-alerta"))],
                 className="titulo-linea"),
        html.A("Ver experimentos en MLflow", href=MLFLOW_UI, target="_blank", rel="noopener",
               className="boton-sec"),
    ], className="titulo")

    kpis = html.Div([
        html.Div([html.Div("Error típico del precio estimado", className="kpi-etq"),
                  html.Div(pct(EVG["mdape_nuevos"], 1), className="kpi-valor"),
                  html.Div(f"Con ubicación · sin ubicación {pct(EV['mdape_nuevos'], 1)} · "
                           f"objetivo {pct(META, 0)} o menos", className="kpi-det")],
                 className="lider-kpi" + (" ok" if vigente else "")),
        html.Div([html.Div("Precios reales dentro del rango sugerido", className="kpi-etq"),
                  html.Div(pct(EVG["cobertura_nuevos"], 0), className="kpi-valor"),
                  html.Div(f"Con ubicación · sin ubicación {pct(EV['cobertura_nuevos'], 0)} · "
                           "objetivo cerca de 80 %", className="kpi-det")]),
        html.Div([html.Div("Diferencia promedio por noche", className="kpi-etq"),
                  html.Div(cop(EVG["mae_nuevos"]), className="kpi-valor"),
                  html.Div(f"Con ubicación · sin ubicación {cop(EV['mae_nuevos'])}", className="kpi-det")]),
        html.Div([html.Div("Consultas en la calculadora", className="kpi-etq"),
                  html.Div(id="m-consultas", className="kpi-valor"),
                  html.Div(id="m-consultas-det", className="kpi-det")]),
    ], className="tarjeta kpis")

    n_nuevos = num(METRICAS["anuncios_nuevos"])
    explicacion = html.Div([
        html.Div([html.H2("Cómo leer este panel")], className="tarjeta-cab"),
        html.Ul([
            html.Li([html.Strong("Cómo se mide. "),
                     "El modelo se entrena con los anuncios de junio de 2026 y su precio estimado se compara "
                     f"con el precio real de {n_nuevos} alojamientos que se publicaron por primera vez en "
                     "septiembre, es decir, alojamientos que el modelo nunca había visto."]),
            html.Li([html.Strong("Error típico. "),
                     "Qué tanto se aleja el precio estimado del precio real. Un error típico de 18 % quiere "
                     "decir que en la mitad de los alojamientos la diferencia es menor a 18 % (menos de "
                     f"$36.000 en un alojamiento de $200.000). El objetivo del proyecto es {pct(META, 0)} o menos."]),
            html.Li([html.Strong("Límite aceptable. "),
                     f"Si el error típico pasa de {pct(UMBRAL, 0)}, el modelo debe actualizarse con datos más "
                     "recientes."]),
            html.Li([html.Strong("Rango sugerido. "),
                     "Es el rango de precios que la calculadora le muestra al anfitrión. Se busca que cerca de "
                     "8 de cada 10 precios reales queden dentro de él."]),
        ], className="explica"),
    ], className="tarjeta", style={"marginBottom": "var(--gap)"})

    error_loc = html.Div([
        html.Div([
            html.Div([html.H2("Error por localidad"), selector_modelo("m-sel-loc")], className="cab-selector"),
            html.P(f"Error típico en los alojamientos nuevos de cada localidad; junto al nombre, cuántos hay. "
                   f"Azul: cumple el objetivo ({pct(META, 0)} o menos). Amarillo: entre {pct(META, 0)} y "
                   f"{pct(UMBRAL, 0)}. Rojo: más de {pct(UMBRAL, 0)}, el precio estimado es menos confiable."),
        ], className="tarjeta-cab"),
        html.Div(id="m-localidad", className="barras"),
    ], className="tarjeta")

    dispersion = html.Div([
        html.Div([
            html.Div([html.H2("Precio estimado frente al publicado"), selector_modelo("m-sel-disp")],
                     className="cab-selector"),
            html.P(f"Cada punto es uno de los {n_nuevos} alojamientos nuevos de septiembre. Si cae sobre la "
                   "diagonal, el precio estimado es igual al publicado; mientras más lejos esté, mayor es la "
                   "diferencia."),
        ], className="tarjeta-cab"),
        html.Div(dcc.Graph(id="m-dispersion", config={"displayModeBar": False}), className="tarjeta-cuerpo"),
    ], className="tarjeta")

    versiones = html.Div([
        html.Div([html.H2("Comparación de modelos"),
                  html.P("Error típico de cada modelo probado en el proyecto, con los mismos alojamientos nuevos; "
                         "mientras más baja la barra, mejor. La mediana por localidad y tipo es el punto de partida: "
                         "estimar con el precio más común de cada localidad y tipo de alojamiento.")],
                 className="tarjeta-cab"),
        html.Div(dcc.Graph(figure=fig_versiones(), config={"displayModeBar": False}), className="tarjeta-cuerpo"),
    ], className="tarjeta")

    consultas = html.Div([
        html.Div([
            html.Div([html.H2("Consultas por localidad"),
                      html.P("Precios calculados en la calculadora durante los últimos 30 días.")],
                     className="tarjeta-cab"),
            html.Div(id="m-tabla"),
        ], className="tarjeta"),
        html.Div([
            html.Div([html.H2("Consultas por tipo de alojamiento"),
                      html.P("Mismo periodo. Muestra qué tipo de anfitrión usa más la herramienta.")],
                     className="tarjeta-cab"),
            html.Div(id="m-tipos", className="barras"),
        ], className="tarjeta"),
    ], className="pila")

    return html.Div([
        titulo, kpis, explicacion,
        html.Div([error_loc, dispersion], className="dos fila-mon"),
        html.Div([versiones, consultas], className="dos"),
        dcc.Interval(id="m-intervalo", interval=30_000),
    ])


def barras_localidad(version):
    por_loc = POR_LOC[version]
    e = pd.Series({k: v["mdape"] for k, v in por_loc.items()}).sort_values(ascending=False)
    tope = max(e.max(), UMBRAL)
    filas = []
    for nombre, v in e.items():
        clase = " neg" if v > UMBRAL else " adv" if v > META else ""
        filas.append(html.Div([
            html.Span([loc(nombre), html.Span(f" · {por_loc[nombre]['n']}", className="sutil")]),
            html.Div(html.Div(className="barra" + clase, style={"width": f"{v / tope * 100:.1f}%"})),
            html.Span(pct(v), className="barra-valor"),
        ], className="barra-linea"))
    return filas


def fig_dispersion(version):
    m = DATOS[DATOS["es_nuevo"] == 1]
    estimado = m[f"precio_{version}"]
    f = fig_base(330)
    f.add_trace(go.Scattergl(x=m[PRECIO], y=estimado, mode="markers",
                             marker=dict(size=4, color=AZUL, opacity=0.45),
                             hovertemplate="Publicado $%{x:,.0f}<br>Estimado $%{y:,.0f}<extra></extra>"))
    lim = [float(min(m[PRECIO].min(), estimado.min())), float(max(m[PRECIO].max(), estimado.max()))]
    f.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=TINTA, width=1), hoverinfo="skip"))
    f.update_xaxes(type="log", title_text="Precio publicado por noche (COP)")
    f.update_yaxes(type="log", title_text="Precio estimado por noche (COP)")
    f.update_layout(margin=dict(t=8, l=56, r=8, b=44))
    return f


def fig_versiones():
    v = sorted(METRICAS["versiones"], key=lambda x: x["mdape_nuevos"])   # de menor a mayor error
    nombres, valores = [x["nombre"] for x in v], [x["mdape_nuevos"] for x in v]
    colores = [MAGENTA if y > UMBRAL else "#ffcd54" if y > META else AZUL for y in valores]
    f = fig_base(300)
    f.add_trace(go.Bar(x=nombres, y=valores, marker_color=colores, width=0.5,
                       text=[pct(y, 1) for y in valores], textposition="outside", cliponaxis=False,
                       textfont=dict(color=TINTA), hoverinfo="skip"))
    f.add_hline(y=UMBRAL, line=dict(color=MAGENTA, width=1.5, dash="dash"))
    f.add_hline(y=META, line=dict(color=GRIS, width=1, dash="dot"))
    f.add_annotation(x=0, xref="paper", y=UMBRAL, text=f"Límite aceptable {pct(UMBRAL, 0)}", showarrow=False,
                     xanchor="left", yanchor="bottom", font=dict(color="#b8004a"))
    f.add_annotation(x=0, xref="paper", y=META, text=f"Objetivo {pct(META, 0)}", showarrow=False,
                     xanchor="left", yanchor="bottom")
    f.update_yaxes(range=[0, max(35, max(valores) + 5)], ticksuffix=" %", title_text="Error típico")
    f.update_layout(margin=dict(t=8, l=56, r=8, b=32))
    return f


# --- Aplicación ---------------------------------------------------------------------------------
app = dash.Dash(__name__, title="BogoPrice", suppress_callback_exceptions=True)
server = app.server
app.layout = html.Div([dcc.Location(id="url"), html.Div(id="marco")])


@app.callback(Output("marco", "children"), Input("url", "pathname"))
def enrutar(ruta):
    ruta = ruta if ruta in ("/", "/monitoreo") else "/"
    contenido = pantalla_monitoreo() if ruta == "/monitoreo" else pantalla_calculadora()
    return [barra_movil(ruta), html.Div([lateral(ruta), html.Main([
        contenido,
        html.Footer("Grupo 6 · Despliegue de Soluciones Analíticas · MIAD, Universidad de los Andes · "
                    "Datos de Inside Airbnb (CC BY 4.0)", className="pie"),
    ], className="lienzo")], className="app")]


@app.callback(Output("f-tipologia", "options"), Output("f-tipologia", "value"), Input("f-tipo", "value"))
def tipologias_del_tipo(tipo):
    opciones_ = opciones_tipologia(tipo)
    valores = [o["value"] for o in opciones_]
    return opciones_, TIPOLOGIA_COMUN[tipo] if TIPOLOGIA_COMUN[tipo] in valores else valores[0]


@app.callback(Output("f-mapa-caja", "style"), Input("f-geo", "value"))
def mostrar_mapa(geo):
    return {} if geo else {"display": "none"}


@app.callback(Output("f-coords", "data"), Output("f-mapa", "figure"),
              Input("f-localidad", "value"), Input("f-mapa", "clickData"), prevent_initial_call=True)
def mover_punto(localidad, click):
    if dash.ctx.triggered_id == "f-mapa" and click:
        punto = click["points"][0]
        lat, lon = punto["lat"], punto["lon"]
    else:
        lat, lon = CENTROS.loc[localidad, "latitude"], CENTROS.loc[localidad, "longitude"]
    return {"lat": float(lat), "lon": float(lon)}, fig_mapa(lat, lon)


@app.callback(
    Output("r-hallazgo", "children"), Output("r-etq", "children"), Output("r-precio", "children"),
    Output("r-rango", "children"), Output("r-escala", "children"),
    Output("r-mediana", "children"), Output("r-mediana-det", "children"),
    Output("r-n", "children"), Output("r-n-det", "children"),
    Output("r-loc", "children"), Output("r-loc-det", "children"),
    Output("r-histograma", "figure"), Output("r-contribuciones", "children"), Output("r-error", "children"),
    Input("f-boton", "n_clicks"),
    State("f-localidad", "value"), State("f-tipo", "value"), State("f-tipologia", "value"),
    State("f-huespedes", "value"), State("f-habitaciones", "value"), State("f-camas", "value"),
    State("f-banos", "value"), State("f-noches", "value"), State("f-compartido", "value"),
    State("f-amenidades", "value"), State("f-geo", "value"), State("f-coords", "data"),
    running=[(Output("f-boton", "disabled"), True, False),
             (Output("f-boton", "children"), "Calculando…", "Calcular precio")],
)
def estimar(n_clicks, localidad, tipo, tipologia, huespedes, habitaciones, camas, banos, noches,
            compartido, amenidades, geo, coords):
    huespedes = int(huespedes or 1)
    anuncio = {
        "neighbourhood_cleansed": localidad,
        "room_type": tipo,
        "property_type": tipologia,
        "accommodates": huespedes,
        "minimum_nights": noches or 1,
        "amenities": [NOMBRE_AIRBNB[k] for k in (amenidades or [])],
    }
    if habitaciones is not None:
        anuncio["bedrooms"] = habitaciones
    if camas is not None:
        anuncio["beds"] = camas
    if banos is not None:
        anuncio["bathrooms_text"] = f"{banos:g} {'shared ' if compartido else ''}bath{'' if banos == 1 else 's'}"
    usar_coordenadas = bool(geo)
    if usar_coordenadas:
        anuncio["latitude"], anuncio["longitude"] = coords["lat"], coords["lon"]

    try:
        r = predecir(anuncio, usar_coordenadas)
        c = contribuciones(anuncio, usar_coordenadas)
    except Exception as e:  # localidad no habilitada, API caída, etc.
        aviso = html.Div(f"No fue posible calcular el precio. Detalle: {e}", className="tarjeta vacio")
        return ("", "", "Sin precio", "", [], "", "", "", "", "", "", fig_base(260), [], aviso)
    if n_clicks:
        registrar_consulta(anuncio, r)

    # Comparables: misma localidad y tipo, con un huésped más o menos
    misma = DATOS[(DATOS["neighbourhood_cleansed"] == localidad) & (DATOS["room_type"] == tipo)]
    comp = misma[misma["accommodates"].between(huespedes - 1, huespedes + 1)]
    desc = f"{PLURAL[tipo].capitalize()} para {max(huespedes - 1, 1)} a {huespedes + 1} huéspedes"
    if len(comp) < 30:
        comp, desc = misma, f"{PLURAL[tipo].capitalize()} de cualquier capacidad (hay pocos parecidos)"
    precios = comp[PRECIO]

    hues = "huésped" if huespedes == 1 else "huéspedes"
    hallazgo = f"{TIPOS[tipo]} para {huespedes} {hues} en {loc(localidad)}: {cop(r['precio'])} por noche."
    if len(comp):
        dif = (r["precio"] / precios.median() - 1) * 100
        hallazgo += (f" Está {pct(abs(dif), 0)} {'por encima' if dif >= 0 else 'por debajo'} del precio típico "
                     f"de {num(len(comp))} {PLURAL[tipo]} parecidos de la misma localidad.")
    etiqueta = "Precio de referencia por noche · " + ("con ubicación exacta" if usar_coordenadas
                                                      else "sin ubicación exacta")
    rango = (f"Rango sugerido: de {cop(r['inferior'])} a {cop(r['superior'])}. Alrededor de 8 de cada 10 "
             "alojamientos parecidos cobran un precio dentro de este rango.")

    h = fig_base(385)
    if len(comp):
        p97 = precios.quantile(0.97)
        h.add_trace(go.Histogram(x=precios[precios <= p97] / 1000, nbinsx=32, marker_color="#cdd8e5",
                                 hovertemplate="%{x} mil: %{y} anuncios<extra></extra>"))
    h.add_vrect(x0=r["inferior"] / 1000, x1=r["superior"] / 1000, fillcolor=AZUL_200, opacity=0.45,
                line_width=0, layer="below")
    h.add_vline(x=r["precio"] / 1000, line=dict(color=AZUL, width=2.5))
    h.add_annotation(x=r["precio"] / 1000, y=1, yref="paper", text="Tu precio", showarrow=False,
                     xanchor="left", yanchor="top", xshift=6, font=dict(color="#0a4f99", size=12))
    h.update_xaxes(title_text="Miles de pesos por noche")
    h.update_yaxes(title_text="Anuncios")
    h.update_layout(bargap=0.06, margin=dict(t=8, l=48, r=8, b=44))

    mediana_loc = float(misma[PRECIO].median()) if len(misma) else None
    return (
        hallazgo, etiqueta, cop(r["precio"]), rango,
        escala(r, precios) if len(comp) else [],
        cop(precios.median()) if len(comp) else "Sin datos", desc,
        num(len(comp)), f"Anuncios de {loc(localidad)} en septiembre de 2026",
        cop(mediana_loc) if mediana_loc else "Sin datos", f"{num(len(misma))} {PLURAL[tipo]} en {loc(localidad)}",
        h, lista_contribuciones(c), None,
    )


@app.callback(Output("m-consultas", "children"), Output("m-consultas-det", "children"),
              Output("m-tabla", "children"), Output("m-tipos", "children"),
              Input("m-intervalo", "n_intervals"))
def monitoreo(_):
    q = pd.read_csv(CONSULTAS) if CONSULTAS.exists() else pd.DataFrame(columns=["fecha", "localidad", "tipo"])
    if len(q):
        q["fecha"] = pd.to_datetime(q["fecha"])
        q = q[q["fecha"] >= q["fecha"].max() - pd.Timedelta(days=30)]
    if not len(q):
        vacio = html.Div("Aún no hay consultas. Cada vez que alguien calcula un precio, aparece aquí.",
                         className="vacio")
        return "0", "Últimos 30 días", vacio, html.Div("Sin consultas en este periodo.",
                                                        className="vacio")
    conteo = q["localidad"].value_counts()
    filas = [html.Tr([html.Td(f"{i}."), html.Td(loc(nombre)), html.Td(num(n), className="num"),
                      html.Td(pct(n / len(q) * 100, 0), className="num")])
             for i, (nombre, n) in enumerate(conteo.head(6).items(), start=1)]
    tabla = html.Table([html.Thead(html.Tr([html.Th(""), html.Th("Localidad"), html.Th("Consultas", className="num"),
                                            html.Th("Participación", className="num")])),
                        html.Tbody(filas)], className="tabla")
    tipo_top = q["tipo"].value_counts(normalize=True)
    detalle = f"Últimos 30 días · {pct(tipo_top.iloc[0] * 100, 0)} de {PLURAL[tipo_top.index[0]]}"
    tipos = q["tipo"].value_counts().reindex(list(TIPOS)).fillna(0)
    barras = [html.Div([
        html.Span(TIPOS[t]),
        html.Div(html.Div(className="barra", style={"width": f"{n / max(tipos.max(), 1) * 100:.1f}%"})
                 if n else None),
        html.Span(num(n), className="barra-valor"),
    ], className="barra-linea") for t, n in tipos.items()]
    return num(len(q)), detalle, tabla, barras


@app.callback(Output("m-localidad", "children"), Input("m-sel-loc", "value"))
def localidad_por_modelo(version):
    return barras_localidad(version)


@app.callback(Output("m-dispersion", "figure"), Input("m-sel-disp", "value"))
def dispersion_por_modelo(version):
    return fig_dispersion(version)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)
