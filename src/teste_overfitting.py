
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Dados fictícios
dados = pd.DataFrame({
    "promocao": [0, 0, 1, 1, 0, 1, 0, 1],
    "pedidos_ontem": [90, 100, 110, 120, 95, 130, 105, 140],
    "pedidos": [95, 105, 140, 150, 100, 16  0, 110, 170]
})

# Separamos os exemplos
# 6 para treinamento e 2 para teste
dados_treino = dados.iloc[:6]
dados_teste = dados.iloc[6:]

variaveis = ["promocao", "pedidos_ontem"]

X_treino = dados_treino[variaveis]
y_treino = dados_treino["pedidos"]

X_teste = dados_teste[variaveis]
y_teste = dados_teste["pedidos"]

# Criando as árvores
arvore_simples = DecisionTreeRegressor(
    max_depth=1,
    random_state=42
)

arvore_complexa = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

# Treinamento
arvore_simples.fit(X_treino, y_treino)
arvore_complexa.fit(X_treino, y_treino)

# Avaliação das duas árvores
for nome, modelo in [
    ("Árvore simples", arvore_simples),
    ("Árvore complexa", arvore_complexa)
]:
    previsao_treino = modelo.predict(X_treino)
    previsao_teste = modelo.predict(X_teste)

    mae_treino = mean_absolute_error(
        y_treino, previsao_treino
    )

    mae_teste = mean_absolute_error(
        y_teste, previsao_teste
    )

    print(f"\n{nome}")
    print(f"MAE treinamento: {mae_treino:.2f}")
    print(f"MAE teste: {mae_teste:.2f}")

    print("Previsões no teste:", previsao_teste)
    print("Valores reais:", y_teste.to_numpy())
