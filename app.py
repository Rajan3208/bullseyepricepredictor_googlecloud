
import logging
logging.basicConfig(level=logging.INFO)

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import pickle
import os

app = Flask(__name__)

# Load the model and scaler
model_path = os.environ.get('MODEL_PATH', 'model/GOOG_prediction_model.keras')
scaler_path = os.environ.get('SCALER_PATH', 'model/scaler.pkl')

# Load the model
model = tf.keras.models.load_model(model_path)

logging.info(f"Checking if model file exists at: {model_path}")
logging.info(f"Directory contents: {os.listdir('model/')}")

# Load the scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get request data
        data = request.get_json()
        ticker = data.get('ticker', 'GOOG')
        days = int(data.get('days', 1))
        
        # Get last 100 days of data
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=150)  # Get extra days for preparation
        
        # Download the stock data
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df[['Close']]
        
        # Scale the data
        last_100_days = df['Close'].values[-100:].reshape(-1, 1)
        last_100_days_scaled = scaler.transform(last_100_days)
        
        # Create prediction data
        X_test = []
        X_test.append(last_100_days_scaled)
        X_test = np.array(X_test)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Predict next n days
        predictions = []
        current_batch = X_test[0]
        
        for i in range(days):
            # Get prediction for next day
            current_pred = model.predict(current_batch.reshape(1, 100, 1))[0]
            
            # Append prediction
            predictions.append(float(scaler.inverse_transform([[current_pred[0]]])[0][0]))
            
            # Update batch for next prediction
            current_batch = np.append(current_batch[1:], [[current_pred[0]]], axis=0)
        
        # Prepare response
        future_dates = [str(end_date + pd.Timedelta(days=i+1)) for i in range(days)]
        response = {
            'ticker': ticker,
            'predictions': [{'date': date, 'price': price} for date, price in zip(future_dates, predictions)]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
