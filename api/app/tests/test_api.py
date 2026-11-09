def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["modelos"] == ["LightGBM v2", "LightGBM v2_geo"]


def test_opciones(client):
    r = client.get("/api/v1/opciones")
    assert r.status_code == 200
    datos = r.json()
    assert "Chapinero" in datos["localidades"]
    assert "Other" in datos["tipologias"]


def test_predict_sin_coordenadas(client, anuncio):
    r = client.post("/api/v1/predict", json=anuncio)
    assert r.status_code == 200
    p = r.json()
    assert p["modelo"] == "LightGBM v2"
    assert p["rango_inferior"] <= p["precio_referencia"] <= p["rango_superior"]
    assert 30_000 < p["precio_referencia"] < 1_500_000


def test_predict_con_coordenadas(client, anuncio):
    r = client.post("/api/v1/predict", json={**anuncio, "usar_coordenadas": True})
    assert r.status_code == 200
    assert r.json()["modelo"] == "LightGBM v2_geo"


def test_localidad_no_habilitada(client, anuncio):
    r = client.post("/api/v1/predict", json={**anuncio, "neighbourhood_cleansed": "Bosa"})
    assert r.status_code == 422
    assert "localidad no habilitada" in r.json()["detail"]


def test_explicar_suma_al_precio(client, anuncio):
    r = client.post("/api/v1/explicar", json=anuncio)
    assert r.status_code == 200
    e = r.json()
    assert len(e["aportes"]) > 5
    assert e["precio_base_cop"] > 0
