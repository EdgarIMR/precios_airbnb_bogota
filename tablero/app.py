"""Tablero BogoPrice (maqueta de la Entrega 1, figuras 5 y 6).

Vistas:
  1. Calculadora del anfitrión: formulario -> precio de referencia, rango sugerido
     del 80 %, indicadores de comparables, posición frente a comparables y
     contribución de cada variable en COP.
  2. Panel de monitoreo: estado del modelo, consultas registradas, MdAPE fuera de
     tiempo, cobertura del rango, error por localidad, estimado vs. publicado y
     evolución del error por captura.

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

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from bogoprice import config
from servicio import contribuciones, predecir, version_modelo

AQUI = Path(__file__).resolve().parent
DATOS = pd.read_csv(AQUI / "datos" / "anuncios_tablero.csv.gz")
METRICAS = json.loads((AQUI / "datos" / "metricas_modelo.json").read_text())
CONSULTAS = Path(os.getenv("CONSULTAS_PATH", AQUI / "datos" / "consultas.csv"))

TIPOS = {
    "Entire home/apt": "Apartamento completo",
    "Private room": "Habitación privada",
    "Hotel room": "Habitación de hotel",
    "Shared room": "Habitación compartida",
}
DATOS["tipo"] = DATOS["room_type"].map(TIPOS)
LOCALIDADES = sorted(DATOS["neighbourhood_cleansed"].unique())
AMENIDADES = {
    "tiene_wifi": "Wi-Fi",
    "tiene_parqueadero": "Parqueadero",
    "tiene_ascensor": "Ascensor",
    "tiene_zona_trabajo": "Zona de trabajo",
    "tiene_agua_caliente": "Agua caliente",
    "tiene_jacuzzi": "Jacuzzi",
}
NOCHES = [1, 2, 3, 4, 7, 14, 30]
AZUL, VERDE, NARANJA, ROJO, GRIS = "#1e3a5f", "#3aa597", "#f2a33a", "#e0525a", "#c9d3df"
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def cop(valor: float) -> str:
    return f"${valor:,.0f}".replace(",", ".")


def num(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", ".")


def pct(valor: float) -> str:
    return f"{valor:.1f} %".replace(".", ",")


def pct2(valor: float) -> str:
    return f"{valor:.2f} %".replace(".", ",")


def campo(etiqueta, control, ayuda=None):
    hijos = [html.Label(etiqueta, className="etiqueta"), control]
    if ayuda:
        hijos.append(html.Div(ayuda, className="ayuda"))
    return html.Div(hijos, className="campo")


def numero(id_, valor, minimo, maximo, paso=1):
    return dcc.Input(id=id_, type="number", value=valor, min=minimo, max=maximo, step=paso,
                     className="entrada")


def tarjeta_kpi(titulo, valor, detalle=None, id_valor=None, id_detalle=None):
    return html.Div([
        html.Div(titulo, className="kpi-titulo"),
        html.Div(valor, className="kpi-valor", id=id_valor) if id_valor else html.Div(valor, className="kpi-valor"),
        html.Div(detalle, className="kpi-detalle", id=id_detalle) if id_detalle else html.Div(detalle, className="kpi-detalle"),
    ], className="kpi")


def grafico(id_):
    return dcc.Graph(id=id_, config={"displayModeBar": False})


def estilo_fig(fig, titulo=None, alto=300):
    fig.update_layout(template="simple_white", height=alto, margin=dict(t=10, l=10, r=10, b=10),
                      font=dict(size=11), showlegend=False)
    return fig


# --- Vista 1: calculadora del anfitrión -------------------------------------------------
formulario = html.Div([
    html.H3("Características de tu alojamiento"),
    campo("Localidad", dcc.Dropdown(LOCALIDADES, "Chapinero", id="f-localidad", clearable=False),
          f"Solo se incluyen localidades con al menos {config.MIN_ANUNCIOS_LOCALIDAD} anuncios para "
          "garantizar estimaciones confiables."),
    campo("Tipo de alojamiento", dcc.Dropdown([{"label": v, "value": k} for k, v in TIPOS.items()],
                                              "Entire home/apt", id="f-tipo", clearable=False)),
    html.Div([
        campo("Huéspedes", numero("f-huespedes", 2, 1, 16)),
        campo("Habitaciones", numero("f-habitaciones", 1, 0, 10)),
        campo("Camas", numero("f-camas", 1, 0, 16)),
        campo("Baños", numero("f-banos", 1, 0, 8, 0.5)),
    ], className="fila"),
    dcc.Checklist([{"label": "Baño compartido", "value": 1}], [], id="f-compartido",
                  className="interruptor"),
    campo("Noches mínimas",
          dcc.RadioItems([{"label": "30 o más" if n == 30 else str(n), "value": n} for n in NOCHES], 2,
                         id="f-noches", className="botonera", inputClassName="oculto"),
          "Los anuncios de 28 noches o más suelen incluir descuento mensual."),
    campo("Amenidades", dcc.Checklist([{"label": v, "value": k} for k, v in AMENIDADES.items()],
                                      ["tiene_wifi", "tiene_ascensor", "tiene_zona_trabajo",
                                       "tiene_agua_caliente"], id="f-amenidades", className="chequeo")),
    html.Button("Calcular precio de referencia", id="f-boton", n_clicks=0, className="boton"),
], className="tarjeta formulario")

resultado = html.Div([
    html.H2("Resultado de la estimación"),
    html.Div(f"Modelo {version_modelo()} · Datos: Inside Airbnb Bogotá, captura de "
             f"{METRICAS['captura_evaluacion'].lower()}", className="subtitulo"),
    html.Div([
        html.Div([
            html.Div("Precio de referencia", className="kpi-titulo"),
            html.Div([html.Span(id="r-precio", className="precio"), html.Span(" / noche", className="noche")]),
            html.Div(id="r-rango", className="rango"),
            html.Div(id="r-base", className="ayuda"),
            html.Div("Este es un precio de mercado de referencia, no una garantía de reservas. "
                     "Úsalo como punto de partida para fijar tu tarifa.", className="aviso"),
        ], className="tarjeta principal"),
        html.Div([
            tarjeta_kpi("Mediana de comparables", "", id_valor="r-mediana", id_detalle="r-mediana-det"),
            tarjeta_kpi("Anuncios similares", "", id_valor="r-similares-n", id_detalle="r-similares-det"),
            tarjeta_kpi("Precio mediano en la localidad", "", id_valor="r-localidad", id_detalle="r-localidad-det"),
        ], className="columna-kpi"),
    ], className="fila-resultado"),
    html.Div([
        html.Div([html.H4("¿Dónde queda tu precio frente a los comparables?"), grafico("r-histograma")],
                 className="tarjeta"),
        html.Div([html.H4("Variables que más pesan en esta estimación"), grafico("r-contribuciones"),
                  html.Div("Contribución de cada variable a la estimación frente a un alojamiento promedio "
                           "del entrenamiento; no representa un efecto causal.", className="ayuda")],
                 className="tarjeta"),
    ], className="dos-columnas-iguales"),
], className="resultado")

vista_calculadora = html.Div([formulario, resultado], className="dos-columnas")

# --- Vista 2: panel de monitoreo ---------------------------------------------------------
ev, en = METRICAS["evaluacion"], METRICAS["entrenamiento"]
vigente = ev["mdape"] <= METRICAS["umbral_reentrenamiento"]
anio, mes, dia = METRICAS["fecha_captura_evaluacion"].split("-")
banner = html.Div([
    html.Div("✓" if vigente else "!", className="banner-icono"),
    html.Div([
        html.Div(f"Estado del modelo: {'VIGENTE' if vigente else 'REQUIERE REENTRENAMIENTO'}",
                 className="banner-titulo"),
        html.Div(
            f"El error en la última captura ({pct2(ev['mdape'])}) está "
            f"{'por debajo' if vigente else 'por encima'} del umbral de reentrenamiento "
            f"({pct(METRICAS['umbral_reentrenamiento'])}). Meta del proyecto: MdAPE ≤ "
            f"{pct(METRICAS['meta_mdape'])}.", className="banner-texto"),
    ]),
], className="banner " + ("banner-ok" if vigente else "banner-alerta"))

vista_monitoreo = html.Div([
    html.H2("Panel de monitoreo del modelo"),
    html.Div(f"Modelo {version_modelo()} · Entrenado con captura de {METRICAS['captura_entrenamiento'].lower()} "
             f"· Evaluado con captura de {METRICAS['captura_evaluacion'].lower()}", className="subtitulo"),
    banner,
    html.Div([
        tarjeta_kpi("Consultas realizadas", "", id_valor="m-consultas", id_detalle="m-consultas-det"),
        tarjeta_kpi("MdAPE en la última captura", pct2(ev["mdape"]),
                    f"Meta ≤ {pct(METRICAS['meta_mdape'])} · Umbral {pct(METRICAS['umbral_reentrenamiento'])}"),
        tarjeta_kpi("Precios reales dentro del rango", pct(ev["cobertura_rango"]), "Meta ≈ 80 %"),
        tarjeta_kpi("Última captura evaluada", f"{dia}-{MESES[int(mes) - 1]}-{anio}",
                    f"{num(METRICAS['anuncios_evaluacion'])} anuncios"),
    ], className="kpis"),
    html.Div([
        html.Div([html.H4("Consultas por localidad"), grafico("m-consultas-localidad")], className="tarjeta"),
        html.Div([html.H4("Consultas por tipo de alojamiento"), grafico("m-consultas-tipo")], className="tarjeta"),
        html.Div([html.H4(f"Precio estimado vs. precio publicado ({METRICAS['captura_evaluacion'].lower()})"),
                  grafico("m-dispersion")], className="tarjeta"),
        html.Div([html.H4("Error (MdAPE) por localidad · línea verde: meta 20 % · línea roja: umbral 25 %"), grafico("m-error-localidad")], className="tarjeta"),
    ], className="dos-columnas-iguales"),
    html.Div([html.H4("Evolución del error por captura"), grafico("m-evolucion")], className="tarjeta"),
    dcc.Interval(id="m-intervalo", interval=30_000),
])

app = dash.Dash(__name__, title="BogoPrice", suppress_callback_exceptions=True)
server = app.server
app.layout = html.Div([
    html.Div([
        html.Div("📈 BogoPrice", className="marca"),
        dcc.Tabs(id="pestanas", value="calculadora", children=[
            dcc.Tab(label="Calculadora del anfitrión", value="calculadora", className="pestana",
                    selected_className="pestana-activa"),
            dcc.Tab(label="Panel de monitoreo", value="monitoreo", className="pestana",
                    selected_className="pestana-activa"),
        ], className="pestanas", parent_className="pestanas-padre"),
        html.Div("MIAD · Proyecto Académico · Grupo 6", className="marca-derecha"),
    ], className="barra"),
    html.Div(vista_calculadora, id="vista-calculadora", className="contenedor"),
    html.Div(vista_monitoreo, id="vista-monitoreo", className="contenedor", style={"display": "none"}),
    html.Footer("© 2026 MIAD · Universidad de los Andes · Despliegue de Soluciones Analíticas · Grupo 6",
                className="pie"),
])


@app.callback(Output("vista-calculadora", "style"), Output("vista-monitoreo", "style"),
              Input("pestanas", "value"))
def cambiar_vista(vista):
    ocultar = {"display": "none"}
    return ({}, ocultar) if vista == "calculadora" else (ocultar, {})


# --- Callbacks de la calculadora ------------------------------------------------------------
def registrar_consulta(anuncio, r):
    nuevo = not CONSULTAS.exists()
    CONSULTAS.parent.mkdir(parents=True, exist_ok=True)
    with CONSULTAS.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha", "localidad", "tipo", "huespedes", "precio_estimado"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), anuncio["neighbourhood_cleansed"],
                    anuncio["room_type"], anuncio["accommodates"], round(r["precio"])])


@app.callback(
    Output("r-precio", "children"), Output("r-rango", "children"), Output("r-base", "children"),
    Output("r-mediana", "children"), Output("r-mediana-det", "children"),
    Output("r-similares-n", "children"), Output("r-similares-det", "children"),
    Output("r-localidad", "children"), Output("r-localidad-det", "children"),
    Output("r-histograma", "figure"), Output("r-contribuciones", "figure"),
    Input("f-boton", "n_clicks"),
    State("f-localidad", "value"), State("f-tipo", "value"), State("f-huespedes", "value"),
    State("f-habitaciones", "value"), State("f-camas", "value"), State("f-banos", "value"),
    State("f-noches", "value"), State("f-compartido", "value"), State("f-amenidades", "value"),
)
def estimar(n_clicks, localidad, tipo, huespedes, habitaciones, camas, banos, noches, compartido, amenidades):
    huespedes = int(huespedes or 1)
    anuncio = {
        "neighbourhood_cleansed": localidad,
        "room_type": tipo,
        "accommodates": huespedes,
        "bedrooms": habitaciones if habitaciones is not None else np.nan,
        "beds": camas if camas is not None else np.nan,
        "bathrooms_count": banos if banos is not None else np.nan,
        "minimum_nights": noches or 1,
        "is_shared_bathroom": int(bool(compartido)),
        **{k: int(k in (amenidades or [])) for k in AMENIDADES},
    }
    r = predecir(anuncio)
    if n_clicks:
        registrar_consulta(anuncio, r)

    # Comparables: misma localidad y tipo, 1 a 3 huéspedes alrededor del valor ingresado
    misma = DATOS[(DATOS["neighbourhood_cleansed"] == localidad) & (DATOS["room_type"] == tipo)]
    comp = misma[misma["accommodates"].between(huespedes - 1, huespedes + 1)]
    desc = f"{TIPOS[tipo].lower()}s de {max(huespedes - 1, 1)} a {huespedes + 1} huéspedes en {localidad}"
    if len(comp) < 30:
        comp, desc = misma, f"{TIPOS[tipo].lower()}s en {localidad}"
    precios = comp[config.TARGET]

    rango = (f"Rango sugerido (intervalo de predicción del 80 %): {cop(r['inferior'])} – {cop(r['superior'])}"
             if r["inferior"] is not None else "")
    base = f"Estimación basada en el comportamiento de {num(len(comp))} alojamientos comparables ({desc})."
    loc_tipo = misma[config.TARGET]

    # Histograma de posición relativa
    h = go.Figure()
    if len(comp):
        h.add_histogram(x=precios[precios <= precios.quantile(0.97)] / 1000, nbinsx=30, marker_color=GRIS)
    if r["inferior"] is not None:
        h.add_vrect(x0=r["inferior"] / 1000, x1=r["superior"] / 1000, fillcolor=VERDE, opacity=0.18, line_width=0)
    h.add_vline(x=r["precio"] / 1000, line_dash="dot", line_color=AZUL, line_width=2,
                annotation_text="Tu precio de referencia", annotation_position="top")
    estilo_fig(h).update_layout(xaxis_title="Miles de COP", yaxis_title="Anuncios", bargap=0.05,
                                margin=dict(t=28, l=10, r=10, b=10))

    # Contribución de variables (las 6 de mayor magnitud)
    c = contribuciones(anuncio)
    c = c.reindex(c["cop"].abs().sort_values(ascending=False).index).head(6).iloc[::-1]
    c["etiqueta"] = c["variable"] + np.where(c["cop"] >= 0, " (+)", " (−)")
    fc = go.Figure(go.Bar(x=c["cop"], y=c["etiqueta"], orientation="h",
                          marker_color=np.where(c["cop"] >= 0, VERDE, ROJO),
                          text=[("+" if v >= 0 else "−") + cop(abs(v)) for v in c["cop"]],
                          textposition="inside", insidetextanchor="middle", textfont_color="white"))
    estilo_fig(fc).update_layout(xaxis_title="Contribución en COP", xaxis_tickformat=",.0f")

    return (cop(r["precio"]), rango, base,
            cop(precios.median()) if len(comp) else "—", f"{num(len(comp))} comparables",
            num(len(comp)), desc.capitalize(),
            cop(loc_tipo.median()) if len(misma) else "—", f"{TIPOS[tipo]} · {localidad}",
            h, fc)


# --- Callbacks del panel de monitoreo -----------------------------------------------------------
def leer_consultas() -> pd.DataFrame:
    if CONSULTAS.exists():
        return pd.read_csv(CONSULTAS)
    return pd.DataFrame(columns=["fecha", "localidad", "tipo", "huespedes", "precio_estimado"])


def figura_vacia(texto):
    f = go.Figure()
    f.add_annotation(text=texto, showarrow=False, font=dict(color="#8a94a0"))
    f.update_xaxes(visible=False)
    f.update_yaxes(visible=False)
    return estilo_fig(f)


@app.callback(
    Output("m-consultas", "children"), Output("m-consultas-det", "children"),
    Output("m-consultas-localidad", "figure"), Output("m-consultas-tipo", "figure"),
    Output("m-dispersion", "figure"), Output("m-error-localidad", "figure"), Output("m-evolucion", "figure"),
    Input("pestanas", "value"), Input("m-intervalo", "n_intervals"),
)
def monitoreo(vista, _):
    q = leer_consultas()
    if len(q):
        q["fecha"] = pd.to_datetime(q["fecha"])
        q = q[q["fecha"] >= q["fecha"].max() - pd.Timedelta(days=30)]
        pl = q["localidad"].value_counts().sort_values()
        f_loc = estilo_fig(go.Figure(go.Bar(x=pl.values, y=pl.index, orientation="h", marker_color=AZUL)))
        pt = q["tipo"].map(TIPOS).value_counts(normalize=True).reindex(TIPOS.values()).fillna(0) * 100
        f_tipo = estilo_fig(go.Figure(go.Bar(x=pt.index, y=pt.values, marker_color=VERDE,
                                             text=[f"{v:.0f} %" for v in pt.values], textposition="outside")))
        f_tipo.update_layout(yaxis_title="% de consultas")
    else:
        f_loc = f_tipo = figura_vacia("Aún no hay consultas registradas")

    # Estimado vs. publicado (muestra para que el gráfico sea liviano)
    m = DATOS.sample(min(len(DATOS), 3000), random_state=0)
    disp = go.Figure(go.Scattergl(x=m[config.TARGET], y=m["precio_estimado"], mode="markers",
                                  marker=dict(size=4, color=AZUL, opacity=0.35)))
    lim = [DATOS[config.TARGET].quantile(0.005), DATOS[config.TARGET].max()]
    disp.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=ROJO, dash="dash")))
    estilo_fig(disp).update_layout(xaxis=dict(type="log", title="Precio publicado (COP)"),
                                   yaxis=dict(type="log", title="Precio estimado (COP)"))

    # MdAPE por localidad
    e = pd.Series(METRICAS["mdape_por_localidad"]).sort_values()
    colores = [VERDE if v <= METRICAS["meta_mdape"] else NARANJA if v <= METRICAS["umbral_reentrenamiento"]
               else ROJO for v in e.values]
    f_err = go.Figure(go.Bar(x=e.index, y=e.values, marker_color=colores,
                             text=[f"{v:.1f}".replace(".", ",") for v in e.values], textposition="outside"))
    f_err.add_hline(y=METRICAS["meta_mdape"], line_dash="dash", line_color=VERDE, line_width=2, layer="above")
    f_err.add_hline(y=METRICAS["umbral_reentrenamiento"], line_dash="dash", line_color=ROJO, line_width=2, layer="above")
    estilo_fig(f_err).update_layout(yaxis_title="MdAPE (%)",
                                    yaxis_range=[0, max(30, e.max() + 3)])

    # Evolución del error por captura
    capt = [METRICAS["captura_entrenamiento"] + " (entrenamiento)", METRICAS["captura_evaluacion"] + " (evaluación)"]
    vals = [en["mdape"], ev["mdape"]]
    evo = go.Figure(go.Scatter(x=capt, y=vals, mode="lines+markers+text", line=dict(color=AZUL),
                               text=[pct2(v) for v in vals], textposition="top center"))
    evo.add_hline(y=METRICAS["umbral_reentrenamiento"], line_dash="dot", line_color=ROJO, line_width=2,
                  annotation_text="Umbral de reentrenamiento (25 %)", annotation_position="top right")
    evo.add_hline(y=METRICAS["meta_mdape"], line_dash="dot", line_color=VERDE, line_width=2, annotation_text="Meta (20 %)",
                  annotation_position="top right")
    estilo_fig(evo, alto=260).update_layout(yaxis_title="MdAPE (%)", yaxis_range=[15, 30],
                                       xaxis_range=[-0.3, 1.3])

    detalle = "Últimos 30 días" if len(q) else "Se registran desde la calculadora"
    return num(len(q)), detalle, f_loc, f_tipo, disp, f_err, evo


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
