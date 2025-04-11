import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
import tensorflow as tf

def train_stock_prediction_model(stock='GOOG', start='2015-01-01', end='2025-12-31', save_path='model'):
    # Download data
    df = yf.download(stock, start=start, end=end)
    df = df.reset_index()
    df = df.drop(['Date', 'Adj Close'], axis=1, errors='ignore')
    
    # Prepare training and testing data
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)
    
    # Prepare sequences for LSTM
    x_train = []
    y_train = []
    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    
    # Build the model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))
    
    # Compile and train the model
    model.compile(optimizer='adam', loss="mse")
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    # Save the Keras model
    keras_model_path = f'{save_path}/{stock}_prediction_model.keras'
    model.save(keras_model_path)
    print(f"Keras model saved to {keras_model_path}")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable resource variables to potentially resolve constant folding issue
    converter.experimental_enable_resource_variables = True
    # Enable 'SELECT_TF_OPS' to support operations not in the default set
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    # Disable experimental tensor list lowering, as suggested in the error message
    converter._experimental_lower_tensor_list_ops = False
    
    tflite_model = converter.convert()
    tflite_model_path = f'{save_path}/{stock}_prediction_model.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {tflite_model_path}")
    
    # Save scaler for inference
    import pickle
    with open(f'{save_path}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler

if __name__ == "__main__":
    train_stock_prediction_model()