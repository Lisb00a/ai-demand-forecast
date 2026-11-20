
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor

from src.train import (
    CAMINHO_DADOS,
    COLUNAS_PT,
    adicionar_variaveis,
)


def validar_modelo():
    dados = pd.read_csv(CAMINHO_DADOS)
    dados = dados.rename(columns=COLUNAS_PT)
    dados = adicionar_variaveis(dados)

    variaveis = [
        "promocao",
        "feriado",
        "dia_da_semana",
        "mes",
        "dia_do_mes",
        "fim_de_semana",
        "pedidos_lag_1",
        "pedidos_lag_7",
        "media_movel_7",
    ]

    # Reservar os 20% finais para teste.
    indice_divisao = int(len(dados) * 0.8)
    dados_treino = dados.iloc[:indice_divisao]

    X = dados_treino[variaveis]
    y = dados_treino["pedidos"]

    tscv = TimeSeriesSplit(n_splits=3)
    resultados = []

    for fold, (idx_treino, idx_validacao) in enumerate(
        tscv.split(X), start=1
    ):
        X_treino = X.iloc[idx_treino]
        X_validacao = X.iloc[idx_validacao]
        y_treino = y.iloc[idx_treino]
        y_validacao = y.iloc[idx_validacao]

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(X_treino, y_treino)

        modelo = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42,
        )
        modelo.fit(X_treino, y_treino)

        previsoes = modelo.predict(X_validacao)
        previsoes_baseline = baseline.predict(X_validacao)

        mae_modelo = mean_absolute_error(
            y_validacao, previsoes
        )
        mae_baseline = mean_absolute_error(
            y_validacao, previsoes_baseline
        )

        resultados.append({
            "fold": fold,
            "mae_modelo": mae_modelo,
            "mae_baseline": mae_baseline,
        })

        print(
            f"Fold {fold} | "
            f"Random Forest: {mae_modelo:.2f} | "
            f"Baseline: {mae_baseline:.2f}"
        )

    resultados_df = pd.DataFrame(resultados)

    print("\n=== RESULTADO DA VALIDACAO TEMPORAL ===")
    print(resultados_df)

    print(
        "\nMAE medio Random Forest:",
        round(resultados_df["mae_modelo"].mean(), 2),
    )
    print(
        "MAE medio Baseline:",
        round(resultados_df["mae_baseline"].mean(), 2),
    )


if __name__ == "__main__":
    validar_modelo()
