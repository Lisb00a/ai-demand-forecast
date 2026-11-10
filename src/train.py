"""
Projeto de previsão de demanda com Machine Learning.
Os dados utilizados são sintéticos e destinados ao aprendizado.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit


# =====================================
# 1. CONFIGURAÇÕES DO PROJETO
# =====================================

PASTA_RAIZ = Path(__file__).resolve().parents[1]

CAMINHO_DADOS = PASTA_RAIZ / "data" / "orders_daily.csv"

PASTA_SAIDA = PASTA_RAIZ / "outputs"

PASTA_SAIDA.mkdir(exist_ok=True)

# Traduz as colunas do CSV (inglês) para português
COLUNAS_PT = {
    "date": "data",
    "orders": "pedidos",
    "promotion": "promocao",
    "holiday": "feriado",
}

NOMES_DIAS = {
    0: "Segunda",
    1: "Terça",
    2: "Quarta",
    3: "Quinta",
    4: "Sexta",
    5: "Sábado",
    6: "Domingo",
}

# Nomes amigáveis das variáveis para os gráficos
NOMES_VARIAVEIS = {
    "promocao": "Promoção",
    "feriado": "Feriado",
    "dia_da_semana": "Dia da semana",
    "mes": "Mês",
    "dia_do_mes": "Dia do mês",
    "fim_de_semana": "Fim de semana",
    "pedidos_lag_1": "Pedidos (1 dia antes)",
    "pedidos_lag_7": "Pedidos (7 dias antes)",
    "media_movel_7": "Média móvel (7 dias)",
}


# =====================================
# 2. CRIAÇÃO DE VARIÁVEIS
# =====================================

def adicionar_variaveis(dados: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis de calendário e histórico de demanda."""

    dados = dados.copy()

    dados["data"] = pd.to_datetime(dados["data"])

    dados = dados.sort_values("data")

    # Variáveis de calendário
    dados["dia_da_semana"] = dados["data"].dt.day_of_week
    dados["mes"] = dados["data"].dt.month
    dados["dia_do_mes"] = dados["data"].dt.day

    dados["fim_de_semana"] = (
        dados["dia_da_semana"] >= 5
    ).astype(int)

    # Variáveis de histórico de demanda
    dados["pedidos_lag_1"] = dados["pedidos"].shift(1)

    dados["pedidos_lag_7"] = dados["pedidos"].shift(7)

    dados["media_movel_7"] = (
        dados["pedidos"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    # Remover linhas sem histórico suficiente
    dados = dados.dropna().reset_index(drop=True)

    return dados


# =====================================
# 3. EXECUÇÃO PRINCIPAL
# =====================================

def executar_projeto():

    # Carregar os dados e traduzir as colunas
    dados = pd.read_csv(CAMINHO_DADOS)

    dados = dados.rename(columns=COLUNAS_PT)

    dados = adicionar_variaveis(dados)

    # =====================================
    # 4. ANÁLISE DOS DADOS
    # =====================================

    print("\n=====================================")
    print("PROJETO: PREVISÃO DE DEMANDA")
    print("=====================================")

    print("\n--- VISÃO GERAL DOS DADOS ---")
    print(dados.head())

    print("\n--- RESUMO ESTATÍSTICO ---")
    print(dados.describe())

    print("\n--- MÉDIA DE PEDIDOS POR PROMOÇÃO (0 = não, 1 = sim) ---")
    print(dados.groupby("promocao")["pedidos"].mean())

    print("\n--- MÉDIA DE PEDIDOS POR FERIADO (0 = não, 1 = sim) ---")
    print(dados.groupby("feriado")["pedidos"].mean())

    print("\n--- MÉDIA DE PEDIDOS POR DIA DA SEMANA ---")
    media_por_dia = dados.groupby("dia_da_semana")["pedidos"].mean()
    media_por_dia.index = media_por_dia.index.map(NOMES_DIAS)
    print(media_por_dia)

    # =====================================
    # 5. SEPARAÇÃO DOS DADOS
    # =====================================

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

    alvo = "pedidos"

    indice_divisao = int(len(dados) * 0.8)

    dados_treino = dados.iloc[:indice_divisao]

    dados_teste = dados.iloc[indice_divisao:]

    print("\n--- COMPARAÇÃO ENTRE TREINO E TESTE ---")

    print(
        "Média de pedidos no treino:",
        round(dados_treino[alvo].mean(), 2)
    )

    print(
        "Média de pedidos no teste:",
        round(dados_teste[alvo].mean(), 2)
    )

    X_treino = dados_treino[variaveis]
    y_treino = dados_treino[alvo]

    X_teste = dados_teste[variaveis]
    y_teste = dados_teste[alvo]

    
    # =====================================
    # 5.1 VALIDAÇÃO TEMPORAL
    # =====================================

    print("\n--- VALIDAÇÃO TEMPORAL ---")

    validacao = TimeSeriesSplit(n_splits=3)

    erros_validacao = []

    for rodada, (indices_treino, indices_validacao) in enumerate(
        validacao.split(X_treino), start=1
    ):

        X_treino_rodada = X_treino.iloc[indices_treino]
        y_treino_rodada = y_treino.iloc[indices_treino]

        X_validacao = X_treino.iloc[indices_validacao]
        y_validacao = y_treino.iloc[indices_validacao]

        modelo_temporal = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42
        )

        modelo_temporal.fit(
            X_treino_rodada,
            y_treino_rodada
        )

        previsoes_validacao = modelo_temporal.predict(
            X_validacao
        )

        mae = mean_absolute_error(
            y_validacao,
            previsoes_validacao
        )

        erros_validacao.append(mae)

        print(f"Rodada {rodada}: MAE = {mae:.2f}")

    print(
        f"MAE médio da validação: "
        f"{sum(erros_validacao) / len(erros_validacao):.2f}"
    )

    # =====================================
    # 5.2 COMPARAÇÃO DE PROFUNDIDADES
    # =====================================
    print("\n--- COMPARAÇÃO DE PROFUNDIDADES ---")

    for profundidade in [3, 5, 8, 12]:
        erros = []

        for indices_treino, indices_validacao in TimeSeriesSplit(
            n_splits=3
        ).split(X_treino):

            modelo_teste = RandomForestRegressor(
                n_estimators=200,
                max_depth=profundidade,
                random_state=42
            )

            modelo_teste.fit(
                X_treino.iloc[indices_treino],
                y_treino.iloc[indices_treino]
            )

            previsoes = modelo_teste.predict(
                X_treino.iloc[indices_validacao]
            )

            mae = mean_absolute_error(
                y_treino.iloc[indices_validacao],
                previsoes
            )

            erros.append(mae)

        media = sum(erros) / len(erros)

        print(
            f"max_depth={profundidade} | "
            f"MAE médio={media:.2f}"
        )


    # =====================================
    # 6. MODELO DE REFERÊNCIA
    # =====================================

    modelo_referencia = DummyRegressor(strategy="mean")

    modelo_referencia.fit(X_treino, y_treino)

    previsoes_referencia = modelo_referencia.predict(X_teste)

    # =====================================
    # 7. MODELO RANDOM FOREST
    # =====================================

    modelo = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )

    modelo.fit(X_treino, y_treino)

    # Salvar o modelo e a ordem das variáveis
    PASTA_MODELOS = PASTA_RAIZ / "models"
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "modelo": modelo,
            "variaveis": variaveis,
        },
        PASTA_MODELOS / "random_forest.joblib"
    )

    print("Modelo salvo com sucesso!")

    print("Profundidade do modelo principal:", modelo.max_depth)

    previsoes = modelo.predict(X_teste)

    # =====================================
    # ANÁLISE DE RESÍDUOS
    # =====================================

    analise_erros = X_teste.copy()

    analise_erros["real"] = y_teste.to_numpy()
    analise_erros["previsto"] = previsoes

    analise_erros["residuo"] = (
        analise_erros["real"] - analise_erros["previsto"]
    )

    analise_erros["erro_absoluto"] = (
        analise_erros["residuo"].abs()
    )

    print("\n--- ANÁLISE DE RESÍDUOS ---")

    print("\nOs 5 maiores erros:")
    print(
        analise_erros.nlargest(
            5, "erro_absoluto"
        )[["real", "previsto", "residuo", "erro_absoluto"]]
    )

    print("\nMAE por promoção:")
    print(
        analise_erros.groupby("promocao")["erro_absoluto"].mean()
    )

    print("\nMAE por dia da semana:")
    print(
        analise_erros.groupby("dia_da_semana")["erro_absoluto"].mean()
    )

    # =====================================
    # ANÁLISE DE VIÉS
    # =====================================

    print("\n--- ANÁLISE DE VIÉS ---")

    print(
        "Resíduo médio:",
        round(analise_erros["residuo"].mean(), 2)
    )

    print("\nResíduo médio por dia da semana:")
    print(
        analise_erros.groupby("dia_da_semana")["residuo"].mean().round(2)
    )

    print("\nResíduo médio por promoção:")
    print(
        analise_erros.groupby("promocao")["residuo"].mean().round(2)
    )
    
    # =====================================
    # VIÉS POR DIA E PROMOÇÃO
    # =====================================

    print("\n--- VIÉS POR DIA E PROMOÇÃO ---")

    tabela_vies = analise_erros.pivot_table(
        index="dia_da_semana",
        columns="promocao",
        values="residuo",
        aggfunc="mean"
    )

    tabela_vies.index = tabela_vies.index.map(NOMES_DIAS)

    print(tabela_vies.round(2))

    print("\n--- QUANTIDADE DE DIAS POR GRUPO ---")

    quantidade_dias = analise_erros.groupby(
        ["dia_da_semana", "promocao"]
    ).size().unstack(fill_value=0)

    quantidade_dias.index = quantidade_dias.index.map(NOMES_DIAS)

    print(quantidade_dias)

    # =====================================
    # SALVAR ANÁLISE DE ERROS
    # =====================================

    analise_erros.to_csv(
        PASTA_SAIDA / "analise_erros.csv",
        index=False
    )

    tabela_vies.to_csv(
        PASTA_SAIDA / "vies_dia_promocao.csv"
    )

    quantidade_dias.to_csv(
        PASTA_SAIDA / "quantidade_dias_grupo.csv"
    )

    print("\nArquivos de diagnóstico salvos!")

    # =====================================
    # 8.1 IMPORTÂNCIA POR PERMUTAÇÃO
    # =====================================

    resultado_permutacao = permutation_importance(
        modelo,
        X_teste,
        y_teste,
        n_repeats=20,
        random_state=42,
        scoring="neg_mean_absolute_error"
    )

    importancias_permutacao = pd.DataFrame({
        "variavel": variaveis,
        "importancia": resultado_permutacao.importances_mean
    })

    importancias_permutacao = importancias_permutacao.sort_values(
        by="importancia",
        ascending=False
    )

    print("\n--- IMPORTÂNCIA POR PERMUTAÇÃO ---")
    print(importancias_permutacao)

    plt.figure(figsize=(10, 6))

    plt.barh(
        importancias_permutacao["variavel"].map(NOMES_VARIAVEIS),
        importancias_permutacao["importancia"]
    )

    plt.title("Importância das variáveis por permutação")
    plt.xlabel("Redução do desempenho quando embaralhada")
    plt.ylabel("Variáveis")

    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA / "importancia_permutacao.png",
        dpi=150
    )

    plt.close()


    # =====================================
    # 8. IMPORTÂNCIA DAS VARIÁVEIS
    # =====================================

    importancias = pd.DataFrame({
        "variavel": variaveis,
        "importancia": modelo.feature_importances_
    })

    importancias = importancias.sort_values(
        by="importancia",
        ascending=False
    )

    print("\n--- IMPORTÂNCIA DAS VARIÁVEIS ---")
    print(importancias)

    plt.figure(figsize=(10, 6))

    plt.barh(
        importancias["variavel"].map(NOMES_VARIAVEIS),
        importancias["importancia"]
    )

    plt.title("Importância das variáveis - Random Forest")
    plt.xlabel("Importância")
    plt.ylabel("Variáveis")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA / "importancia_variaveis.png",
        dpi=150
    )

    plt.close()

    # =====================================
    # 9. AVALIAÇÃO DOS MODELOS
    # =====================================

    erro_referencia = mean_absolute_error(
        y_teste,
        previsoes_referencia
    )

    erro_modelo = mean_absolute_error(
        y_teste,
        previsoes
    )

    raiz_erro_quadratico = mean_squared_error(
        y_teste,
        previsoes
    ) ** 0.5

    print("\n=====================================")
    print("RESULTADOS DOS MODELOS")
    print("=====================================")

    print(f"Erro médio - Referência: {erro_referencia:.2f} pedidos")

    print(f"Erro médio - Random Forest: {erro_modelo:.2f} pedidos")

    print(
        f"Raiz do erro quadrático - Random Forest: "
        f"{raiz_erro_quadratico:.2f} pedidos"
    )

    # =====================================
    # 10. SALVAR PREVISÕES
    # =====================================

    resultados = dados_teste[["data", alvo]].copy()

    resultados["pedidos_previstos"] = previsoes.round(1)

    resultados.to_csv(
        PASTA_SAIDA / "previsoes.csv",
        index=False
    )

    # =====================================
    # 11. GERAR GRÁFICO
    # =====================================

    plt.figure(figsize=(12, 5))

    plt.plot(
        resultados["data"],
        resultados["pedidos"],
        label="Pedidos reais"
    )

    plt.plot(
        resultados["data"],
        resultados["pedidos_previstos"],
        label="Pedidos previstos"
    )

    plt.title("Previsão de demanda: pedidos reais x previstos")
    plt.xlabel("Data")
    plt.ylabel("Quantidade de pedidos")

    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA / "grafico_previsao.png",
        dpi=150
    )

    plt.close()

    # =====================================
    # 12. FINALIZAÇÃO
    # =====================================

    print("\n--- ARQUIVOS GERADOS ---")

    print(f"Previsões: {PASTA_SAIDA / 'previsoes.csv'}")

    print(f"Gráfico: {PASTA_SAIDA / 'grafico_previsao.png'}")

    print(
        f"Importância das variáveis: "
        f"{PASTA_SAIDA / 'importancia_variaveis.png'}"
    )

    # NOVO: Mostrar onde foi salvo o gráfico de permutação
    print(
        f"Importância por permutação: "
        f"{PASTA_SAIDA / 'importancia_permutacao.png'}"
    )

    print("\nProjeto executado com sucesso!")

if __name__ == "__main__":
    executar_projeto()
