## Resultados do Modelo

O projeto utiliza dados sintéticos para prever a demanda diária de pedidos, comparando o desempenho de um modelo **Random Forest Regressor** com um modelo de referência (**DummyRegressor**).

### Metodologia de Avaliação

Os dados foram divididos cronologicamente, respeitando a ordem temporal das observações:

- **80% dos dados:** treinamento e validação temporal.
- **20% dos dados:** teste final, reservado para avaliar o desempenho do modelo.

A validação temporal foi realizada utilizando `TimeSeriesSplit`, com três divisões, permitindo avaliar o modelo em diferentes períodos sem utilizar observações futuras no treinamento.

Também foram comparadas diferentes profundidades de árvores para identificar a configuração com menor erro médio na validação.

### Comparação de Desempenho

| Métrica | DummyRegressor | Random Forest |
|---|---:|---:|
| MAE no teste final | 11,59 | 8,83 |
| MAE médio na validação temporal | 15,73 | 10,40 |
| RMSE no teste final | — | 10,79 |

O Random Forest apresentou uma **redução de aproximadamente 23,8% no erro absoluto médio (MAE)** em comparação com o modelo de referência no conjunto de teste.

### Resultados da Validação Temporal

| Rodada | MAE Random Forest | MAE Baseline |
|---|---:|---:|
| 1 | 11,93 | 14,38 |
| 2 | 9,17 | 15,16 |
| 3 | 10,11 | 17,65 |

O Random Forest apresentou menor erro em todas as rodadas da validação temporal.

### Comparação de Hiperparâmetros

Foram avaliadas quatro configurações de profundidade máxima:

| Profundidade máxima | MAE médio |
|---|---:|
| 3 | 11,17 |
| 5 | 10,40 |
| 8 | 10,79 |
| 12 | 10,89 |

A configuração