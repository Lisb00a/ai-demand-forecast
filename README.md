# AI Demand Forecast — Machine Learning Demand Prediction

An end-to-end Machine Learning project for forecasting daily order demand using historical data, calendar features, promotions, and holidays.

Built with **Python, Scikit-learn, Streamlit, FastAPI, and GitHub Actions**, this project demonstrates the development of a predictive model, temporal validation, interactive visualization, REST API integration, and automated testing.

> **Disclaimer:** This project uses synthetic data for educational and portfolio purposes. Results do not represent real-world production performance.

## Project Overview

Accurate demand forecasting helps businesses anticipate order volumes, optimize resource allocation, and support operational planning.

This project explores how supervised Machine Learning can estimate daily orders using historical demand patterns and contextual features.

The solution includes:

- Exploratory data analysis and feature engineering
- Random Forest regression
- Baseline comparison using DummyRegressor
- Chronological train/test split
- Time-series cross-validation
- Hyperparameter comparison
- Residual and bias analysis
- Interactive Streamlit dashboard
- FastAPI prediction endpoint
- Automated tests with pytest
- Continuous integration with GitHub Actions

## Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| REST API | FastAPI, Uvicorn |
| Model Serialization | Joblib |
| Testing | Pytest, HTTPX |
| CI/CD | GitHub Actions |
| Version Control | Git, GitHub |

## Dataset

The project uses a synthetic dataset containing daily order information.

Original data fields include:

- `date`: observation date
- `orders`: daily order volume
- `promotion`: promotion indicator
- `holiday`: holiday indicator
- `revenue_brl`: daily revenue in Brazilian reais

The dataset contains approximately two years of daily observations, from January 2024 through December 2025.

### Feature Engineering

The model uses nine input features:

| Feature | Description |
|---|---|
| `promocao` | Promotion indicator |
| `feriado` | Holiday indicator |
| `dia_da_semana` | Day of the week |
| `mes` | Month |
| `dia_do_mes` | Day of the month |
| `fim_de_semana` | Weekend indicator |
| `pedidos_lag_1` | Orders from the previous day |
| `pedidos_lag_7` | Orders from seven days earlier |
| `media_movel_7` | Average orders over the previous seven days |

Historical features are calculated using past observations to avoid directly including the target day's order volume.

The first seven observations are excluded because they lack sufficient historical information.

## Machine Learning Methodology

The workflow follows these steps:

1. Load and preprocess the synthetic dataset.
2. Generate calendar and historical demand features.
3. Split