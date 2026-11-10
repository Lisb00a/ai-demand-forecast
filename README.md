
# AI Demand Forecast — Previsão de Demanda

## Sobre o projeto

Projeto de Machine Learning desenvolvido em Python para prever a quantidade diária de pedidos com base em dados históricos, promoções, feriados e características do calendário.

Os dados são sintéticos e utilizados para fins educacionais.

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Git e GitHub

## Metodologia

1. Preparação e exploração dos dados.
2. Engenharia de variáveis temporais.
3. Separação cronológica entre treino e teste.
4. Treinamento de um modelo de referência (DummyRegressor).
5. Treinamento de Random Forest.
6. Validação temporal com TimeSeriesSplit.
7. Comparação de hiperparâmetros.
8. Avaliação com MAE e RMSE.
9. Análise de resíduos e viés.

## Resultados

| Modelo | MAE |
|---|---:|
| DummyRegressor | 11.59 |
| Random Forest | 8.83 |

O Random Forest apresentou uma redução de aproximadamente 23,8% no MAE em relação ao modelo de referência.

**Configuração selecionada:**
- n_estimators: 200
- max_depth: 5
- random_state: 42

**RMSE no teste:** 10.79 pedidos.

## Principais descobertas

- A variável de promoção apresentou grande importância nas previsões.
- O modelo apresentou resíduo médio positivo de 2,14 pedidos.
- Sextas-feiras sem promoção apresentaram subestimação média de 8,53 pedidos no conjunto de teste.
- Grupos com poucas observações precisam ser interpretados com cautela.

## Como executar

Instale as dependências do projeto e execute:

```bash
python src/train.py
```

Os arquivos de previsões, gráficos e análises são gerados na pasta `outputs/`.

## Próximas melhorias

- Implementar testes automatizados com pytest.
- Organizar o código em módulos reutilizáveis.
- Experimentar outros modelos utilizando validação temporal.
- Criar uma API de previsão com FastAPI.

## Resultados da Validação Temporal

O modelo foi avaliado com TimeSeriesSplit
utilizando cinco períodos de validação.

| Modelo | MAE médio |
|--------|-----------|
| Random Forest | 10.53 |
| DummyRegressor | 14.75 |

O Random Forest apresentou redução aproximada
de 28.6% no MAE médio em comparação ao baseline.

O modelo superou o baseline nos cinco períodos
avaliados.

Os dados utilizados são sintéticos e os
resultados não representam desempenho em produção.