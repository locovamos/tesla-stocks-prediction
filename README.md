# TSLA Stock Price Prediction Using CNN-LSTM

This repository contains a Jupyter Notebook that utilizes a Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) network to predict Tesla's (TSLA) stock closing price. The project includes downloading historical stock data, preprocessing, and training the deep learning model to make predictions.

## Project Overview

The project follows these steps:

1. **Data Collection**:
   - The notebook uses `yfinance` to download Tesla stock data and saves it as a CSV file for further analysis.

2. **Data Preprocessing**:
   - The data is cleaned and filtered to include only relevant columns like `Open`, `High`, `Low`, and `Close`.
   - The dataset is scaled using the `MinMaxScaler` to normalize values between 0 and 1.
   - The dataset is split into training, validation, and test sets with a lookback window of 30 days to feed into the model.

3. **Model Architecture**:
   - The notebook implements a **CNN-LSTM model**:
     - **CNN** layers are used to capture local patterns in the time series data.
     - **LSTM** layers help the model learn from long-term dependencies in the stock price trends.
   - The model is compiled using the Adam optimizer and the Mean Squared Error (MSE) loss function.

4. **Training and Evaluation**:
   - The model is trained using the training dataset, and its performance is evaluated on validation and test sets.
   - Performance metrics such as **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and **R-squared** are used to evaluate the model's accuracy on the test data.

5. **Predictions and Visualization**:
   - The trained model is used to predict Tesla's stock closing price on the test dataset.
   - A plot is generated to compare the **Actual** vs. **Predicted** stock prices.

## Dependencies

To run this notebook, you will need the following libraries:

- `yfinance`: for fetching historical stock data.
- `pandas`: for data manipulation.
- `numpy`: for numerical computations.
- `matplotlib`: for visualizing the results.
- `scikit-learn`: for data scaling and metrics evaluation.
- `tensorflow` or `keras`: for building and training the CNN-LSTM model.

Install the necessary dependencies with:
```bash
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow
