
import matplotlib

# Salva imagens sem abrir janelas
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Dados fictícios para aprender
dados = pd.DataFrame({
    "promocao": [0, 0, 1, 1, 0, 1, 0, 1],
    "pedidos_ontem": [90, 100, 110, 120, 95, 130, 105, 140],
    "pedidos": [95, 105, 140, 150, 100, 160, 110, 170]
})

# Informações que o modelo vai analisar
X = dados[["promocao", "pedidos_ontem"]]

# O que queremos prever
y = dados["pedidos"]

# Criando a árvore
modelo = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

# Treinando
modelo.fit(X, y)

# Fazendo uma previsão
novo_dia = pd.DataFrame({
    "promocao": [1],
    "pedidos_ontem": [110]
})

previsao = modelo.predict(novo_dia)

print("Previsão de pedidos:", previsao[0])

# Criando a pasta de saída
Path("outputs").mkdir(exist_ok=True)

# Desenhando a árvore
plt.figure(figsize=(14, 8))

plot_tree(
    modelo,
    feature_names=["promocao", "pedidos_ontem"],
    filled=True,
    rounded=True
)

# Salvando a imagem
plt.savefig("outputs/arvore_decisao.png", dpi=140)
plt.close()

print("Árvore salva em outputs/arvore_decisao.png")
