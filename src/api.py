from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest.joblib"

app = FastAPI(
    title="AI Demand Forecast API",
    version="1.0.0"
)


class PedidoPrevisao(BaseModel):
    data: date
    promocao: int = Field(ge=0, le=1)
    feriado: int = Field(ge=0, le=1)
    historico_pedidos: list[int] = Field(min_length=7, max_length=7)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(entrada: PedidoPrevisao):

    if any(p < 0 for p in entrada.historico_pedidos):
        raise HTTPException(
            status_code=422,
            detail="Pedidos não podem ser negativos"
        )

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Modelo não encontrado"
        )

    artefato = joblib.load(MODEL_PATH)
    modelo = artefato["modelo"]
    variaveis = artefato["variaveis"]

    historico = entrada.historico_pedidos
    dia_semana = entrada.data.weekday()

    dados = {
        "promocao": entrada.promocao,
        "feriado": entrada.feriado,
        "dia_da_semana": dia_semana,
        "mes": entrada.data.month,
        "dia_do_mes": entrada.data.day,
        "fim_de_semana": int(dia_semana >= 5),
        "pedidos_lag_1": historico[-1],
        "pedidos_lag_7": historico[0],
        "media_movel_7": sum(historico) / 7,
    }

    X = pd.DataFrame([dados])[variaveis]

    previsao = float(modelo.predict(X)[0])

    return {
        "data": entrada.data.isoformat(),
        "pedidos_previstos": round(max(0, previsao), 1),
        "modelo": "RandomForestRegressor"
    }