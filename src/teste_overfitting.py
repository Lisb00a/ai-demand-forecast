
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Dados fictícios
dados = pd.DataFrame({
    "promocao": [0, 0, 1, 1, 0, 1, 0, 1],
    "pedidos_ontem": [90, 100, 110, 120, 95, 130, 105, 140],
    "pedidos": [95, 105, 140, 150, 100, 160, 110, 170]
})

# Dados para ensinar os modelos
X = dados[["promocao", "pedidos_ontem"]]
y = dados["pedidos"]

# Criando duas árvores diferentes
arvore_simples = DecisionTreeRegressor(
    max_depth=1,
    random_state=42
)

arvore_complexa = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

# Treinando as duas árvores
arvore_simples.fit(X, y)
arvore_complexa.fit(X, y)

# Previsões nos próprios dados de treinamento
previsao_simples = arvore_simples.predict(X)
previsao_complexa = arvore_complexa.predict(X)

# Calculando os erros
mae_simples = mean_absolute_error(y, previsao_simples)
mae_complexa = mean_absolute_error(y, previsao_complexa)

print("================================")
print("COMPARAÇÃO DAS ÁRVORES")
print("================================")

print(f"MAE - Árvore simples: {mae_simples:.2f}")
print(f"MAE - Árvore complexa: {mae_complexa:.2f}")
