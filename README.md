# Stock Price Prediction

This project utilizes historical stock price data and various technical analysis indicators as features to predict future stock `close` prices. 

The model is built using Long Short-Term Memory (LSTM), a type of recurrent neural network suitable for time-series forecasting.

---

## Dataset and Feature Engineering

### Dataset
The dataset is sourced from the Yahoo Finance API and includes historical price data for multiple stocks, including:
- AAPL, AMZN, GOOGL, many more.

### Features
We calculate a variety of technical analysis indicators as features:
1. **Momentum Indicators**
   - Relative Strength Index (RSI)
   - Williams %R
2. **Moving Averages**
   - Simple Moving Averages (SMA): 7, 14, 21 days
   - Exponential Moving Averages (EMA): 7, 25, 99, 200 days
3. **Trend Indicators**
   - Moving Average Convergence Divergence (MACD)
   - Commodity Channel Index (CCI)
4. **Volatility Indicators**
   - Bollinger Bands (Upper and Lower)
   - Ulcer Index

---

## Model Architecture

The model uses an LSTM layer to capture temporal dependencies in the stock price data. It predicts a sequence of 5 future `close` prices based on a 14-day input window. 

### Architecture
- **Input Shape:** `(n_steps_in=14, n_features)`
- **LSTM Layer:** 50 units
- **Dense Layer:** Produces `n_steps_out=5` predictions
- **Optimizer:** Adam with a learning rate of `1e-4`
- **Loss Function:** Mean Absolute Error (MAE)

---

## Evaluation

### Metrics
The model is evaluated on the test set using the following metrics:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

### Visualization
Predictions are visualized by plotting the actual vs. predicted prices for each stock.

---

## Results

### Summary Metrics
Evaluation metrics for all stocks are summarized in a DataFrame. Detailed metrics and prediction plots for each stock are available.

### Example Output
- **AAPL:**
  - MSE: 2.3456
  - RMSE: 1.5312
  - MAE: 1.2345
- ...

---

## How to Use

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run `process_data.py` to prepare the dataset and compute features.
4. Train the model using `stock_prediction_LSTM_temporal.py`.
5. Evaluate and visualize results using the built-in evaluation functions.

---

## Future Improvements
1. Experiment with different architectures (e.g., GAN models).
2. Add sentiment analysis from news articles to complement technical indicators.
3. Improve hyperparameter tuning for better performance.

---