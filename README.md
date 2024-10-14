# Stock Price Prediction using Machine Learning (Linear Regression and LSTM)

## Overview
This project demonstrates how to predict future stock prices using machine learning models like **Linear Regression** and deep learning models such as **Long Short-Term Memory (LSTM)**. Stock market prediction is a challenging task due to the inherent randomness of financial markets, but by leveraging historical data, we can create models that attempt to forecast future prices.

## Features
- **Linear Regression** model to predict stock prices based on historical data.
- **LSTM (Long Short-Term Memory)** deep learning model for time series prediction.
- Evaluation of the models using accuracy metrics like RMSE and MAPE.
- Visualizations to compare predicted vs. actual stock prices.

## Libraries Used
- `numpy`: For numerical operations.
- `pandas`: For data manipulation.
- `matplotlib`: For plotting stock prices and predictions.
- `scikit-learn`: For implementing linear regression and scaling data.
- `tensorflow`: For building the LSTM model.

## How to Run the Project
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/stock-price-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd stock-price-prediction
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter Notebook file (`Stock_Price_Prediction.ipynb`) using:
    ```bash
    jupyter notebook
    ```
5. Follow the code cells to train and evaluate the machine learning models.

## Model Evaluation
- The **Linear Regression** model provides a baseline prediction with reasonable accuracy for stock prices.
- The **LSTM** model improves on the linear regression model by capturing trends and dependencies in time-series data, showing lower error rates in predictions.

## Future Enhancements
- Integrating real-time stock data from an API to make live predictions.
- Incorporating additional features like moving averages, trading volume, and technical indicators to improve prediction accuracy.
- Deploying the model on a web app using Flask or Streamlit.

## Results
The results of the prediction are visualized, comparing the predicted stock prices with the actual values. The LSTM model outperforms the linear regression model, particularly for time-series data.



