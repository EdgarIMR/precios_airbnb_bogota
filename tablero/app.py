"""Tablero BogoPrice (maqueta de la Entrega 1, figuras 5 y 6).

Dos pantallas:
  /             Calculadora: el anfitrión describe su alojamiento y recibe el precio
                de referencia, el rango donde se publican 8 de cada 10 alojamientos
                parecidos, su posición frente a los comparables y qué variables mueven
                el precio.
  /monitoreo    Monitoreo del modelo: error fuera de tiempo frente a la meta y al
                umbral de reentrenamiento, cobertura del rango, error por localidad,
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

from bogoprice import config
from servicio import contribuciones, opciones, origen, predecir

AQUI = Path(__file__).resolve().parent
DATOS = pd.read_csv(AQUI / "datos" / "anuncios_tablero.csv.gz")
METRICAS = json.loads((AQUI / "datos" / "metricas_modelo.json").read_text())
CONSULTAS = Path(os.getenv("CONSULTAS_PATH", AQUI / "datos" / "consultas.csv"))
MLFLOW_UI = os.getenv("MLFLOW_UI_URL", "http://3.82.57.253:8050")
OPCIONES = opciones()

TIPOS = {
    "Entire home/apt": "Apartamento completo",
    "Private room": "Habitación privada",
    "Hotel room": "Habitación de hotel",
    "Shared room": "Habitación compartida",
}
PLURAL = {
    "Entire home/apt": "apartamentos completos",
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
COBERTURA_NOMINAL = round((METRICAS["percentiles_rango"][1] - METRICAS["percentiles_rango"][0]) * 100)


def loc(nombre: str) -> str:
    return NOMBRE_LOC.get(nombre, nombre)


def cop(valor: float) -> str:
    return f"${valor:,.0f}".replace(",", ".")


def num(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", ".")


def pct(valor: float, dec: int = 1) -> str:
    return f"{valor:.{dec}f} %".replace(".", ",")


def puntos(valor: float) -> str:
    return f"{valor:.2f}".replace(".", ",") + " puntos"


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
                  f"{len(LOCALIDADES)} localidades con {config.MIN_ANUNCIOS_LOCALIDAD} anuncios o más"],
                 className="ambito"),
        html.Div("Anfitriones", className="grupo"),
        item("/", "Calculadora de precio", ICONO_CALC),
        html.Div("Equipo del proyecto", className="grupo"),
        item("/monitoreo", "Monitoreo del modelo", ICONO_MON),
        html.Div([html.Strong("Modelo LightGBM v2"),
                  f"Entrenado con {num(FINAL['n_anuncios'])} anuncios de junio y septiembre 2026"],
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
        campo("Localidad", dcc.Dropdown([{"label": loc(l), "value": l} for l in LOCALIDADES], "Chapinero",
                                        id="f-localidad", clearable=False),
              f"Solo aparecen las localidades con {config.MIN_ANUNCIOS_LOCALIDAD} anuncios o más; "
              "en las demás no hay datos suficientes para estimar con confianza."),
        dcc.Checklist([{"label": "Marcar la ubicación exacta en el mapa", "value": 1}], [], id="f-geo",
                      className="casillas"),
        html.Div([
            dcc.Graph(id="f-mapa", figure=fig_mapa(centro.latitude, centro.longitude),
                      config={"displayModeBar": False, "scrollZoom": True}),
            html.Div("Haz clic donde queda tu alojamiento. Con la ubicación exacta el modelo se equivoca "
                     f"menos: {pct(EVG['mdape_nuevos'])} frente a {pct(EV['mdape_nuevos'])}.", className="ayuda"),
        ], id="f-mapa-caja", className="mapa-caja", style={"display": "none"}),
        dcc.Store(id="f-coords", data={"lat": centro.latitude, "lon": centro.longitude}),
        campo("Tipo de alojamiento", dcc.Dropdown([{"label": v, "value": k} for k, v in TIPOS.items()],
                                                  "Entire home/apt", id="f-tipo", clearable=False)),
        campo("Tipo de inmueble", dcc.Dropdown(opciones_tipologia("Entire home/apt"), "Entire rental unit",
                                               id="f-tipologia", clearable=False)),
        html.Div([
            campo("Huéspedes", numero("f-huespedes", 2, 1, 16), para="f-huespedes"),
            campo("Habitaciones", numero("f-habitaciones", 1, 0, 10), para="f-habitaciones"),
            campo("Camas", numero("f-camas", 1, 0, 16), para="f-camas"),
            campo("Baños", numero("f-banos", 1, 0, 8, 0.5), para="f-banos"),
        ], className="par"),
        campo("Noches mínimas",
              dcc.RadioItems([{"label": "30+" if n == 30 else str(n), "value": n} for n in NOCHES], 2,
                             id="f-noches", className="botonera"),
              "Desde 28 noches Airbnb aplica el descuento mensual, y el precio publicado ya lo incluye."),
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
        html.P("Es un precio de mercado de referencia y no garantiza reservas. Úsalo como punto de partida "
               "para fijar tu tarifa.", className="aviso", style={"marginTop": "4px"}),
        html.Div(id="r-escala", className="escala", role="img",
                 **{"aria-label": "Rango sugerido frente a los comparables"}),
        html.Div([
            html.Div([html.Div("Mediana de los comparables", className="kpi-etq"),
                      html.Div(id="r-mediana", className="kpi-valor"),
                      html.Div(id="r-mediana-det", className="kpi-det")]),
            html.Div([html.Div("Comparables", className="kpi-etq"),
                      html.Div(id="r-n", className="kpi-valor"),
                      html.Div(id="r-n-det", className="kpi-det")]),
            html.Div([html.Div("Mediana del tipo en la localidad", className="kpi-etq"),
                      html.Div(id="r-loc", className="kpi-valor"),
                      html.Div(id="r-loc-det", className="kpi-det")]),
        ], className="franja"),
    ], className="tarjeta lider")

    paneles = html.Div([
        html.Div([
            html.Div([html.H2("Dónde queda tu precio"),
                      html.P("Precio por noche de los comparables; la franja azul es tu rango sugerido.")],
                     className="tarjeta-cab"),
            html.Div(dcc.Graph(id="r-histograma", config={"displayModeBar": False}), className="tarjeta-cuerpo"),
        ], className="tarjeta"),
        html.Div([
            html.Div([html.H2("Qué mueve tu precio"),
                      html.P("Pesos por noche que suma o resta cada variable frente al precio base del modelo. "
                             "Describe el modelo, no un efecto causal.")], className="tarjeta-cab"),
            html.Div(id="r-contribuciones", className="barras"),
        ], className="tarjeta"),
    ], className="dos")

    titulo = html.Div([
        html.Div([html.H1("Calculadora de precio"), html.P(id="r-hallazgo", className="hallazgo")]),
        html.Span(f"Captura del {fecha(METRICAS['fecha_captura_evaluacion'])}", className="chip chip-neutro"),
    ], className="titulo")

    return html.Div([
        titulo,
        html.Div([formulario, html.Div([lider, html.Div(id="r-error"), paneles], className="resultado")],
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
        html.Div(f"Mediana {cop(med)}", className="escala-etq abajo", style={"left": pos(med)}),
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
def pantalla_monitoreo():
    vigente = EV["mdape_nuevos"] <= UMBRAL
    cumple = EV["mdape_nuevos"] <= META
    n_nuevos = num(METRICAS["anuncios_nuevos"])
    por_loc = METRICAS["mdape_por_localidad"]
    sobre = [loc(k) for k, v in sorted(por_loc.items(), key=lambda x: -x[1]["mdape"]) if v["mdape"] > UMBRAL]
    ridge = METRICAS["versiones"][0]["mdape_nuevos"]
    hallazgo = (f"En los {n_nuevos} anuncios nuevos de septiembre el error mediano es {pct(EV['mdape_nuevos'], 2)}"
                f"{', dentro de la meta de ' + pct(META, 0) if cumple else ''}. Con la ubicación exacta baja a "
                f"{pct(EVG['mdape_nuevos'], 2)}; el Ridge anterior tenía {pct(ridge, 2)}.")
    if len(sobre) > 1:
        hallazgo += f" {', '.join(sobre[:-1])} y {sobre[-1]} siguen por encima del umbral."
    elif sobre:
        hallazgo += f" {sobre[0]} sigue por encima del umbral."

    titulo = html.Div([
        html.Div([
            html.Div([html.H1("Monitoreo del modelo"),
                      html.Span("Vigente" if vigente else "Requiere reentrenamiento",
                                className="chip " + ("chip-ok" if vigente else "chip-alerta"))],
                     className="titulo-linea"),
            html.P(hallazgo, className="hallazgo"),
        ]),
        html.A("Ver experimentos en MLflow", href=MLFLOW_UI, target="_blank", rel="noopener",
               className="boton-sec"),
    ], className="titulo")

    kpis = html.Div([
        html.Div([html.Div("Error mediano (MdAPE) en anuncios nuevos", className="kpi-etq"),
                  html.Div(pct(EV["mdape_nuevos"], 2), className="kpi-valor"),
                  html.Div(f"Meta {pct(META, 0)} · umbral {pct(UMBRAL, 0)} · con ubicación "
                           f"{pct(EVG['mdape_nuevos'], 2)}", className="kpi-det")],
                 className="lider-kpi" + (" ok" if vigente else "")),
        html.Div([html.Div("Precios reales dentro del rango", className="kpi-etq"),
                  html.Div(pct(EV["cobertura_nuevos"]), className="kpi-valor"),
                  html.Div(f"Rango entre los percentiles 7,5 y 92,5 (meta cercana a 80 %)", className="kpi-det")]),
        html.Div([html.Div("Error absoluto medio", className="kpi-etq"),
                  html.Div(cop(EV["mae_nuevos"]), className="kpi-valor"),
                  html.Div(f"{cop(EVG['mae_nuevos'])} con ubicación exacta", className="kpi-det")]),
        html.Div([html.Div("Consultas registradas", className="kpi-etq"),
                  html.Div(id="m-consultas", className="kpi-valor"),
                  html.Div(id="m-consultas-det", className="kpi-det")]),
    ], className="tarjeta kpis")

    e = pd.Series({k: v["mdape"] for k, v in por_loc.items()}).sort_values(ascending=False)
    filas = []
    for nombre, v in e.items():
        clase = " neg" if v > UMBRAL else " adv" if v > META else ""
        filas.append(html.Div([
            html.Span([loc(nombre), html.Span(f" · {por_loc[nombre]['n']}", className="sutil")]),
            html.Div(html.Div(className="barra" + clase, style={"width": f"{v / e.max() * 100:.1f}%"})),
            html.Span(pct(v), className="barra-valor"),
        ], className="barra-linea"))
    error_loc = html.Div([
        html.Div([html.H2("Error por localidad"),
                  html.P(f"MdAPE en anuncios nuevos de septiembre (junto al nombre, cuántos son). Magenta: por "
                         f"encima del umbral de {pct(UMBRAL, 0)}. Ámbar: entre la meta de {pct(META, 0)} y el "
                         "umbral.")], className="tarjeta-cab"),
        html.Div(filas, className="barras"),
    ], className="tarjeta")

    consultas = html.Div([
        html.Div([
            html.Div([html.H2("Consultas por localidad"),
                      html.P("Precios calculados por anfitriones en los últimos 30 días.")], className="tarjeta-cab"),
            html.Div(id="m-tabla"),
        ], className="tarjeta"),
        html.Div([
            html.Div([html.H2("Consultas por tipo de alojamiento"),
                      html.P("Mismo periodo. Muestra qué segmento usa más la calculadora.")],
                     className="tarjeta-cab"),
            html.Div(id="m-tipos", className="barras"),
        ], className="tarjeta"),
    ], className="pila")

    dispersion = html.Div([
        html.Div([html.H2("Precio estimado frente al publicado"),
                  html.P(f"Los {n_nuevos} anuncios nuevos de septiembre, con el modelo entrenado solo con junio. "
                         "Escala logarítmica; sobre la diagonal el modelo acierta.")], className="tarjeta-cab"),
        html.Div(dcc.Graph(figure=fig_dispersion(), config={"displayModeBar": False}), className="tarjeta-cuerpo"),
    ], className="tarjeta")

    versiones = html.Div([
        html.Div([html.H2("Error por versión del modelo"),
                  html.P("MdAPE en los mismos anuncios nuevos de septiembre, entrenando con junio.")],
                 className="tarjeta-cab"),
        html.Div(dcc.Graph(figure=fig_versiones(), config={"displayModeBar": False}), className="tarjeta-cuerpo"),
    ], className="tarjeta")

    return html.Div([
        titulo, kpis,
        html.Div([error_loc, consultas], className="dos fila-mon"),
        html.Div([dispersion, versiones], className="dos"),
        dcc.Interval(id="m-intervalo", interval=30_000),
    ])


def fig_dispersion():
    m = DATOS[DATOS["es_nuevo"] == 1]
    f = fig_base(300)
    f.add_trace(go.Scattergl(x=m[config.TARGET], y=m["precio_validacion"], mode="markers",
                             marker=dict(size=4, color=AZUL, opacity=0.45),
                             hovertemplate="Publicado %{x:,.0f}<br>Estimado %{y:,.0f}<extra></extra>"))
    lim = [float(m[config.TARGET].min()), float(m[config.TARGET].max())]
    f.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=TINTA, width=1), hoverinfo="skip"))
    f.update_xaxes(type="log", title_text="Precio publicado (COP)")
    f.update_yaxes(type="log", title_text="Precio estimado (COP)")
    f.update_layout(margin=dict(t=8, l=56, r=8, b=44))
    return f


def fig_versiones():
    v = METRICAS["versiones"]
    nombres, valores = [x["nombre"] for x in v], [x["mdape_nuevos"] for x in v]
    colores = [MAGENTA if y > UMBRAL else "#ffcd54" if y > META else AZUL for y in valores]
    f = fig_base(300)
    f.add_trace(go.Bar(x=nombres, y=valores, marker_color=colores, width=0.5,
                       text=[pct(y, 2) for y in valores], textposition="outside", cliponaxis=False,
                       textfont=dict(color=TINTA), hoverinfo="skip"))
    f.add_hline(y=UMBRAL, line=dict(color=MAGENTA, width=1.5, dash="dash"))
    f.add_hline(y=META, line=dict(color=GRIS, width=1, dash="dot"))
    f.add_annotation(x=1, xref="paper", y=UMBRAL, text=f"Umbral {pct(UMBRAL, 0)}", showarrow=False,
                     xanchor="right", yanchor="bottom", font=dict(color="#b8004a"))
    f.add_annotation(x=1, xref="paper", y=META, text=f"Meta {pct(META, 0)}", showarrow=False,
                     xanchor="right", yanchor="bottom")
    f.update_yaxes(range=[0, 30], ticksuffix=" %", title_text="MdAPE")
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
        html.Footer(f"MIAD · Universidad de los Andes · Despliegue de Soluciones Analíticas · Grupo 6 · "
                    f"Modelo servido por {origen()}", className="pie"),
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
        aviso = html.Div(f"No se pudo calcular el precio: {e}", className="tarjeta vacio")
        return ("", "", "Sin precio", "", [], "", "", "", "", "", "", fig_base(260), [], aviso)
    if n_clicks:
        registrar_consulta(anuncio, r)

    # Comparables: misma localidad y tipo, con un huésped más o menos
    misma = DATOS[(DATOS["neighbourhood_cleansed"] == localidad) & (DATOS["room_type"] == tipo)]
    comp = misma[misma["accommodates"].between(huespedes - 1, huespedes + 1)]
    desc = f"{PLURAL[tipo].capitalize()} de {max(huespedes - 1, 1)} a {huespedes + 1} huéspedes"
    if len(comp) < 30:
        comp, desc = misma, f"{PLURAL[tipo].capitalize()} de cualquier capacidad"
    precios = comp[config.TARGET]

    hues = "huésped" if huespedes == 1 else "huéspedes"
    hallazgo = f"{TIPOS[tipo]} para {huespedes} {hues} en {loc(localidad)}: {cop(r['precio'])} por noche."
    if len(comp):
        dif = (r["precio"] / precios.median() - 1) * 100
        hallazgo += (f" Queda {pct(abs(dif), 0)} {'por encima' if dif >= 0 else 'por debajo'} de la mediana de "
                     f"{num(len(comp))} {PLURAL[tipo]} parecidos en la misma localidad.")
    etiqueta = "Precio de referencia por noche · " + ("con ubicación exacta" if usar_coordenadas else "por localidad")
    rango = (f"Cerca de 8 de cada 10 alojamientos como el tuyo se publican entre {cop(r['inferior'])} y "
             f"{cop(r['superior'])}.")

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

    mediana_loc = float(misma[config.TARGET].median()) if len(misma) else None
    return (
        hallazgo, etiqueta, cop(r["precio"]), rango,
        escala(r, precios) if len(comp) else [],
        cop(precios.median()) if len(comp) else "Sin datos", desc,
        num(len(comp)), f"En {loc(localidad)}, septiembre 2026",
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
        vacio = html.Div("Todavía no hay consultas. Cada precio calculado en la calculadora aparece aquí.",
                         className="vacio")
        return "0", "Se cuentan desde la calculadora", vacio, html.Div("Sin consultas en el periodo.",
                                                                        className="vacio")
    conteo = q["localidad"].value_counts()
    filas = [html.Tr([html.Td(f"{i}."), html.Td(loc(nombre)), html.Td(num(n), className="num"),
                      html.Td(pct(n / len(q) * 100, 0), className="num")])
             for i, (nombre, n) in enumerate(conteo.head(6).items(), start=1)]
    tabla = html.Table([html.Thead(html.Tr([html.Th(""), html.Th("Localidad"), html.Th("Consultas", className="num"),
                                            html.Th("Participación", className="num")])),
                        html.Tbody(filas)], className="tabla")
    tipo_top = q["tipo"].value_counts(normalize=True)
    detalle = f"Últimos 30 días · {pct(tipo_top.iloc[0] * 100, 0)} son {PLURAL[tipo_top.index[0]]}"
    tipos = q["tipo"].value_counts().reindex(list(TIPOS)).fillna(0)
    barras = [html.Div([
        html.Span(TIPOS[t]),
        html.Div(html.Div(className="barra", style={"width": f"{n / max(tipos.max(), 1) * 100:.1f}%"})
                 if n else None),
        html.Span(num(n), className="barra-valor"),
    ], className="barra-linea") for t, n in tipos.items()]
    return num(len(q)), detalle, tabla, barras


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
