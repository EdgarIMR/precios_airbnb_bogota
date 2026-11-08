"""Tablero BogoPrice: precio de referencia por noche para anfitriones en Bogotá.

Dos vistas:
  1. Estimar mi precio: formulario con las variables del modelo servible ->
     precio de referencia, rango competitivo y comparación con anuncios similares.
  2. Mercado: precios de la captura más reciente (septiembre 2026) por localidad
     y tipo de alojamiento, con mapa de anuncios.

Ejecutar desde la raíz del repo:
    python tablero/preparar_tablero.py   # una vez: modelo + datos
    python tablero/app.py                # http://localhost:8050
"""
from __future__ import annotations

from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from bogoprice import config
from servicio import predecir, version_modelo

AQUI = Path(__file__).resolve().parent
DATOS = pd.read_csv(AQUI / "datos" / "anuncios_tablero.csv.gz")

TIPOS = {
    "Entire home/apt": "Alojamiento entero",
    "Private room": "Habitación privada",
    "Hotel room": "Habitación de hotel",
    "Shared room": "Habitación compartida",
}
DATOS["tipo"] = DATOS["room_type"].map(TIPOS)
LOCALIDADES = sorted(DATOS["neighbourhood_cleansed"].unique())
AMENIDADES = {
    "tiene_wifi": "Wifi",
    "tiene_agua_caliente": "Agua caliente",
    "tiene_zona_trabajo": "Zona de trabajo",
    "tiene_parqueadero": "Parqueadero",
    "tiene_ascensor": "Ascensor",
    "tiene_jacuzzi": "Jacuzzi",
}
AZUL, NARANJA, GRIS = "#1f5f8b", "#e07a1f", "#8a94a0"


def cop(valor: float) -> str:
    return f"${valor:,.0f}".replace(",", ".")


def campo(etiqueta: str, control) -> html.Div:
    return html.Div([html.Label(etiqueta, className="etiqueta"), control], className="campo")


def numero(id_: str, valor, minimo, maximo, paso=1):
    return dcc.Input(id=id_, type="number", value=valor, min=minimo, max=maximo, step=paso,
                     className="entrada", debounce=True)


# --- Vista 1: estimar precio -----------------------------------------------------
formulario = html.Div(
    [
        html.H3("Describe tu alojamiento"),
        campo("Localidad", dcc.Dropdown(LOCALIDADES, "Chapinero", id="f-localidad", clearable=False)),
        campo("Tipo de alojamiento", dcc.Dropdown(
            [{"label": v, "value": k} for k, v in TIPOS.items()], "Entire home/apt",
            id="f-tipo", clearable=False)),
        html.Div([
            campo("Huéspedes", numero("f-huespedes", 2, 1, 16)),
            campo("Habitaciones", numero("f-habitaciones", 1, 0, 10)),
            campo("Camas", numero("f-camas", 1, 0, 16)),
            campo("Baños", numero("f-banos", 1, 0, 8, 0.5)),
        ], className="fila"),
        campo("Noches mínimas", numero("f-noches", 2, 1, 365)),
        dcc.Checklist([{"label": " Baño compartido", "value": 1}], [], id="f-compartido",
                      className="chequeo"),
        campo("Amenidades", dcc.Checklist(
            [{"label": " " + v, "value": k} for k, v in AMENIDADES.items()],
            ["tiene_wifi", "tiene_agua_caliente"], id="f-amenidades", className="chequeo")),
        html.Button("Estimar precio", id="f-boton", n_clicks=0, className="boton"),
    ],
    className="tarjeta formulario",
)

resultado = html.Div(
    [
        html.Div([
            html.Div([html.Div("Precio de referencia por noche", className="kpi-titulo"),
                      html.Div(id="r-precio", className="kpi-valor")], className="kpi"),
            html.Div([html.Div("Rango competitivo (P10–P90)", className="kpi-titulo"),
                      html.Div(id="r-rango", className="kpi-valor kpi-chico")], className="kpi"),
            html.Div([html.Div("Ubicación frente a anuncios similares", className="kpi-titulo"),
                      html.Div(id="r-percentil", className="kpi-valor kpi-chico")], className="kpi"),
        ], className="kpis"),
        html.Div(id="r-nota", className="nota"),
        dcc.Graph(id="r-similares", config={"displayModeBar": False}),
    ],
    className="tarjeta resultado",
)

vista_estimar = html.Div([formulario, resultado], className="dos-columnas")

# --- Vista 2: mercado ---------------------------------------------------------------
vista_mercado = html.Div(
    [
        html.Div([
            campo("Localidades", dcc.Dropdown(LOCALIDADES, [], id="m-localidades", multi=True,
                                              placeholder="Todas")),
            campo("Tipo de alojamiento", dcc.Dropdown(
                [{"label": v, "value": k} for k, v in TIPOS.items()], [], id="m-tipos",
                multi=True, placeholder="Todos")),
            campo("Huéspedes", dcc.RangeSlider(1, 16, 1, value=[1, 16], id="m-huespedes",
                                               marks={i: str(i) for i in (1, 4, 8, 12, 16)})),
        ], className="tarjeta filtros"),
        html.Div(id="m-kpis", className="kpis"),
        html.Div([
            html.Div(dcc.Graph(id="m-localidad", config={"displayModeBar": False}), className="tarjeta"),
            html.Div(dcc.Graph(id="m-tipo", config={"displayModeBar": False}), className="tarjeta"),
        ], className="dos-columnas-iguales"),
        html.Div(dcc.Graph(id="m-mapa"), className="tarjeta"),
    ]
)

app = dash.Dash(__name__, title="BogoPrice")
server = app.server
app.layout = html.Div(
    [
        html.Header([
            html.H1("BogoPrice"),
            html.P("Precio de referencia por noche para anfitriones independientes de Airbnb en Bogotá"),
        ], className="encabezado"),
        dcc.Tabs(
            [
                dcc.Tab(vista_estimar, label="Estimar mi precio", className="pestana",
                        selected_className="pestana-activa"),
                dcc.Tab(vista_mercado, label="Mercado por localidad", className="pestana",
                        selected_className="pestana-activa"),
            ],
            className="pestanas",
        ),
        html.Footer(
            f"Datos: Inside Airbnb Bogotá, septiembre 2026 ({len(DATOS):,} anuncios en "
            f"{len(LOCALIDADES)} localidades con ≥{config.MIN_ANUNCIOS_LOCALIDAD} anuncios). "
            f"Modelo: {version_modelo()}. MIAD · Grupo 6.".replace(",", "."),
            className="pie",
        ),
    ],
    className="contenedor",
)


# --- Callbacks -------------------------------------------------------------------------
@app.callback(
    Output("r-precio", "children"), Output("r-rango", "children"),
    Output("r-percentil", "children"), Output("r-nota", "children"),
    Output("r-similares", "figure"),
    Input("f-boton", "n_clicks"),
    State("f-localidad", "value"), State("f-tipo", "value"), State("f-huespedes", "value"),
    State("f-habitaciones", "value"), State("f-camas", "value"), State("f-banos", "value"),
    State("f-noches", "value"), State("f-compartido", "value"), State("f-amenidades", "value"),
)
def estimar(_, localidad, tipo, huespedes, habitaciones, camas, banos, noches, compartido, amenidades):
    anuncio = {
        "neighbourhood_cleansed": localidad,
        "room_type": tipo,
        "accommodates": huespedes or 1,
        "bedrooms": habitaciones if habitaciones is not None else np.nan,
        "beds": camas if camas is not None else np.nan,
        "bathrooms_count": banos if banos is not None else np.nan,
        "minimum_nights": noches or 1,
        "is_shared_bathroom": int(bool(compartido)),
        **{k: int(k in (amenidades or [])) for k in AMENIDADES},
    }
    r = predecir(anuncio)

    # Anuncios similares: misma localidad y tipo, capacidad parecida (±1 huésped)
    sim = DATOS[(DATOS["neighbourhood_cleansed"] == localidad) & (DATOS["room_type"] == tipo)
                & DATOS["accommodates"].between(anuncio["accommodates"] - 1, anuncio["accommodates"] + 1)]
    if len(sim) < 30:  # muestra chica: se amplía a toda la localidad y tipo
        sim = DATOS[(DATOS["neighbourhood_cleansed"] == localidad) & (DATOS["room_type"] == tipo)]
    criterio = f"{TIPOS[tipo].lower()} en {localidad}"

    fig = go.Figure()
    if len(sim):
        tope = sim[config.TARGET].quantile(0.97)
        fig.add_histogram(x=sim.loc[sim[config.TARGET] <= tope, config.TARGET], nbinsx=40,
                          marker_color=GRIS, opacity=0.6, name="Anuncios similares")
    if r["inferior"] is not None:
        fig.add_vrect(x0=r["inferior"], x1=r["superior"], fillcolor=AZUL, opacity=0.12,
                      line_width=0, annotation_text="Rango competitivo",
                      annotation_position="top left")
    fig.add_vline(x=r["precio"], line_color=NARANJA, line_width=3,
                  annotation_text="Tu precio de referencia", annotation_position="top right")
    fig.update_layout(title=f"Precio por noche de anuncios similares ({criterio}, n={len(sim)})",
                      xaxis_title="Precio por noche (COP)", yaxis_title="Anuncios",
                      template="simple_white", showlegend=False, margin=dict(t=60, l=50, r=20, b=50),
                      xaxis_tickformat=",.0f")

    if len(sim):
        pct = (sim[config.TARGET] < r["precio"]).mean() * 100
        percentil = f"Más alto que el {pct:.0f} % de los similares"
        mediana = f"La mediana de los similares es {cop(sim[config.TARGET].median())}."
    else:
        percentil, mediana = "Sin anuncios similares", ""
    rango = f"{cop(r['inferior'])} – {cop(r['superior'])}" if r["inferior"] is not None else "—"
    nota = (f"{mediana} El rango competitivo es el intervalo donde se ubican 8 de cada 10 anuncios "
            f"con estas características; error típico del modelo ±25 % (MdAPE en septiembre 2026).")
    return cop(r["precio"]), rango, percentil, nota, fig


def filtrar(localidades, tipos, huespedes):
    d = DATOS
    if localidades:
        d = d[d["neighbourhood_cleansed"].isin(localidades)]
    if tipos:
        d = d[d["room_type"].isin(tipos)]
    return d[d["accommodates"].between(*huespedes)]


@app.callback(
    Output("m-kpis", "children"), Output("m-localidad", "figure"),
    Output("m-tipo", "figure"), Output("m-mapa", "figure"),
    Input("m-localidades", "value"), Input("m-tipos", "value"), Input("m-huespedes", "value"),
)
def mercado(localidades, tipos, huespedes):
    d = filtrar(localidades, tipos, huespedes)
    p = d[config.TARGET]
    kpis = [
        html.Div([html.Div(t, className="kpi-titulo"), html.Div(v, className="kpi-valor kpi-chico")],
                 className="kpi")
        for t, v in [
            ("Anuncios", f"{len(d):,}".replace(",", ".")),
            ("Precio mediano por noche", cop(p.median()) if len(d) else "—"),
            ("Rango intercuartil (P25–P75)", f"{cop(p.quantile(.25))} – {cop(p.quantile(.75))}" if len(d) else "—"),
            ("Alojamientos enteros", f"{(d['room_type'] == 'Entire home/apt').mean() * 100:.0f} %" if len(d) else "—"),
        ]
    ]

    por_loc = (d.groupby("neighbourhood_cleansed")[config.TARGET]
               .agg(mediana="median", anuncios="size").reset_index().sort_values("mediana"))
    f_loc = px.bar(por_loc, x="mediana", y="neighbourhood_cleansed", orientation="h",
                   text=por_loc["mediana"].map(cop), hover_data={"anuncios": True},
                   labels={"mediana": "Precio mediano por noche (COP)", "neighbourhood_cleansed": ""},
                   title="Precio mediano por localidad", template="simple_white",
                   color_discrete_sequence=[AZUL])
    f_loc.update_traces(textposition="outside", cliponaxis=False)
    f_loc.update_layout(margin=dict(t=50, l=10, r=40, b=40), xaxis_tickformat=",.0f")

    tope = p.quantile(0.97) if len(d) else 0
    f_tipo = px.box(d[p <= tope], x="tipo", y=config.TARGET, points=False,
                    labels={"tipo": "", config.TARGET: "Precio por noche (COP)"},
                    title="Distribución del precio por tipo (hasta P97)",
                    template="simple_white", color_discrete_sequence=[AZUL],
                    category_orders={"tipo": list(TIPOS.values())})
    f_tipo.update_layout(margin=dict(t=50, l=10, r=10, b=40), yaxis_tickformat=",.0f")

    muestra = d.sample(min(len(d), 4000), random_state=0) if len(d) else d
    f_mapa = px.scatter_map(muestra, lat="latitude", lon="longitude", color=config.TARGET,
                            color_continuous_scale="Viridis", range_color=(p.quantile(.05), p.quantile(.95)) if len(d) else None,
                            hover_data={"neighbourhood_cleansed": True, "tipo": True,
                                        config.TARGET: ":,.0f", "latitude": False, "longitude": False},
                            labels={config.TARGET: "Precio (COP)", "neighbourhood_cleansed": "Localidad",
                                    "tipo": "Tipo"},
                            zoom=11, height=520, map_style="carto-positron",
                            title="Ubicación y precio de los anuncios (muestra de hasta 4.000)")
    f_mapa.update_traces(marker=dict(size=6, opacity=0.7))
    f_mapa.update_layout(margin=dict(t=50, l=0, r=0, b=0))
    return kpis, f_loc, f_tipo, f_mapa


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
