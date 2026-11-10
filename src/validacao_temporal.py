from pathlib import Path

import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "orders_daily.csv"


def preparar_dados():

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    # Variáveis de calendário
    df["dia_semana"] = df["date"].dt.day_of_week
    df["mes"] = df["date"].dt.month
    df["final_semana"] = (
        df["dia_semana"] >= 5
    ).astype(int)

    # Histórico de pedidos
    df["pedidos_lag_1"] = df["orders"].shift(1)

    df["media_movel_7"] = (
        df["orders"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    df = df.dropna().reset_index(drop=True)

    return df


def validar_modelo():

    df = preparar_dados()

    features = [
        "promotion",
        "holiday",
        "dia_semana",
        "mes",
        "final_semana",
        "pedidos_lag_1",
        "media_movel_7"
    ]

    X = df[features]
    y = df["orders"]

    tscv = TimeSeriesSplit(n_splits=5)

    resultados = []

    for fold, (train_index, test_index) in enumerate(
        tscv.split(X), start=1
    ):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(X_train, y_train)

        modelo = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )

        modelo.fit(X_train, y_train)

        previsoes = modelo.predict(X_test)
        previsoes_baseline = baseline.predict(X_test)

        mae = mean_absolute_error(y_test, previsoes)
        mae_baseline = mean_absolute_error(
            y_test, previsoes_baseline
        )

        resultados.append({
            "fold": fold,
            "mae_modelo": mae,
            "mae_baseline": mae_baseline
        })

        print(f"\nFold {fold}")
        print(f"MAE Random Forest: {mae:.2f}")
        print(f"MAE Baseline: {mae_baseline:.2f}")

    resultados_df = pd.DataFrame(resultados)

    print("\nRESULTADO FINAL")
    print(resultados_df)

    print(
        "\nMAE médio Random Forest:",
        round(resultados_df["mae_modelo"].mean(), 2)
    )

    print(
        "MAE médio Baseline:",
        round(resultados_df["mae_baseline"].mean(), 2)
    )


if __name__ == "__main__":
    print("Iniciando validação temporal...")
    validar_modelo()