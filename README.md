# Sri Lanka Dengue Forecasting Using Ensemble Learning

## Project Overview
This project predicts weekly dengue cases in Colombo, Sri Lanka, using machine learning, ensemble learning, statistical forecasting, and hybrid techniques.

## Models Implemented
- Random Forest
- Bagging Regressor
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- Artificial Neural Network (ANN)
- Stacking Ensemble
- Random Forest + ANN Hybrid
- ARIMA
- SARIMA
- ARIMA + Random Forest Hybrid

## Technologies
Python, Google Colab, Scikit-learn, XGBoost, LightGBM, CatBoost, Statsmodels, Streamlit, Pandas, NumPy, Matplotlib, and Plotly.

## Model Evaluation
The models were compared using MAE, RMSE, and R².

In the reference experiment, the Random Forest + ANN hybrid achieved an RMSE of 43.68 and an R² score of 0.742.

## Running the Dashboard
1. Install the required packages using `pip install -r requirements.txt`.
2. Run `streamlit run app.py`.
3. Open the local Streamlit URL displayed in the terminal.

## Limitations
This project uses historical data for Colombo. The dataset's original provenance requires verification, and predictions should not be interpreted as official public-health forecasts.