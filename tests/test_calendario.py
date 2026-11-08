
import pandas as pd

from src.train import adicionar_variaveis


def test_media_movel_7():
    dados = pd.DataFrame({
        "data": pd.date_range("2026-01-01", periods=10),
        "pedidos": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
        "promocao": [0] * 10,
        "feriado": [0] * 10
    })

    resultado = adicionar_variaveis(dados)

    assert resultado.iloc[0]["media_movel_7"] == 130
    assert resultado.iloc[1]["media_movel_7"] == 140
    assert resultado.iloc[2]["media_movel_7"] == 150


def test_pedidos_lag_1():
    dados = pd.DataFrame({
        "data": pd.date_range("2026-01-01", periods=10),
        "pedidos": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
        "promocao": [0] * 10,
        "feriado": [0] * 10
    })

    resultado = adicionar_variaveis(dados)

    assert resultado.iloc[0]["pedidos_lag_1"] == 160
    assert resultado.iloc[1]["pedidos_lag_1"] == 170
    assert resultado.iloc[2]["pedidos_lag_1"] == 180


def test_dia_da_semana():
    dados = pd.DataFrame({
        "data": pd.date_range("2026-01-01", periods=10),
        "pedidos": [100] * 10,
        "promocao": [0] * 10,
        "feriado": [0] * 10
    })

    resultado = adicionar_variaveis(dados)

    # A primeira data restante será 08/01/2026 (quinta-feira)
    assert resultado.iloc[0]["dia_da_semana"] == 3
