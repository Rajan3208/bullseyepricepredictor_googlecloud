from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import pickle
import os
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Print startup information
logger.info("=== APPLICATION STARTING ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Directory contents: {os.listdir('.')}")

try:
    logger.info(f"Model directory contents: {os.listdir('model/')}")
except Exception as e:
    logger.error(f"Error checking model directory: {str(e)}")

app = Flask(__name__)

# Define a simple health check route
@app.route('/', methods=['GET'])
def home():
    return "Bullseye Price Predictor is running. Use /predict endpoint for predictions."

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

# Initialize global variables
model = None
scaler = None

# Load models in a function to handle errors properly
def initialize_models():
    global model, scaler
    
    try:
        model_path = os.environ.get('MODEL_PATH', 'model/GOOG_prediction_model.keras')
        scaler_path = os.environ.get('SCALER_PATH', 'model/scaler.pkl')
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Loading scaler from: {scaler_path}")
        
        # Check if files exist
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
            
        if not os.path.exists(scaler_path):
            logger.error(f"Scaler file not found at {scaler_path}")
            return False
        
        # Load TensorFlow model
        logger.info("Attempting to load TensorFlow model...")
        model = tf.keras.models.load_model(model_path)
        logger.info("TensorFlow model loaded successfully")
        
        # Load scaler
        logger.info("Attempting to load scaler...")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Initialize models at startup
models_loaded = initialize_models()
logger.info(f"Models initialized successfully: {models_loaded}")

@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return jsonify({"error": "Models failed to load. Please check server logs."}), 500
        
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        ticker = data.get('ticker', 'GOOG')
        days = int(data.get('days', 1))
        
        logger.info(f"Processing prediction request for {ticker} for {days} days")
        
        # Get last 100 days of data
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=150)
        
        # Download stock data
        logger.info(f"Downloading stock data for {ticker}")
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            return jsonify({"error": f"No data available for ticker {ticker}"}), 404
            
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
        logger.info(f"Making predictions for next {days} days")
        predictions = []
        current_batch = X_test[0]
        
        for i in range(days):
            # Make prediction with verbose=0 to reduce log noise
            current_pred = model.predict(current_batch.reshape(1, 100, 1), verbose=0)[0]
            predictions.append(float(scaler.inverse_transform([[current_pred[0]]])[0][0]))
            current_batch = np.append(current_batch[1:], [[current_pred[0]]], axis=0)
        
        # Prepare response
        future_dates = [str(end_date + pd.Timedelta(days=i+1)) for i in range(days)]
        response = {
            'ticker': ticker,
            'predictions': [{'date': date, 'price': price} for date, price in zip(future_dates, predictions)]
        }
        
        logger.info("Prediction successful")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

logger.info("=== APPLICATION INITIALIZED ===")

if __name__ == '__main__':
    # Get port from environment variable
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
