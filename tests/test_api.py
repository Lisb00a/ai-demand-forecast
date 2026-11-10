
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


# TESTE 1: Verificar se a API está funcionando
def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# TESTE 2: Verificar se o modelo faz previsões
def test_predict():
    dados = {
        "data": "2026-11-10",
        "promocao": 1,
        "feriado": 0,
        "historico_pedidos": [
            110, 120, 115, 130, 125, 140, 135
        ]
    }

    response = client.post("/predict", json=dados)

    assert response.status_code == 200

    resultado = response.json()

    assert "pedidos_previstos" in resultado
    assert isinstance(resultado["pedidos_previstos"], (int, float))
    assert resultado["pedidos_previstos"] >= 0
    assert resultado["modelo"] == "RandomForestRegressor"


# TESTE 3: Rejeitar histórico insuficiente
def test_historico_insuficiente():
    dados = {
        "data": "2026-11-10",
        "promocao": 1,
        "feriado": 0,
        "historico_pedidos": [110, 120]
    }

    response = client.post("/predict", json=dados)

    assert response.status_code == 422


# TESTE 4: Rejeitar promoção inválida
def test_promocao_invalida():
    dados = {
        "data": "2026-11-10",
        "promocao": 5,
        "feriado": 0,
        "historico_pedidos": [
            110, 120, 115, 130, 125, 140, 135
        ]
    }

    response = client.post("/predict", json=dados)

    assert response.status_code == 422
