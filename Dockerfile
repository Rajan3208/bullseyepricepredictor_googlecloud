FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install flask==2.0.1 werkzeug==2.0.1 gunicorn==20.1.0
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model and application code
COPY model/ ./model/
COPY app.py .

# Set environment variables
ENV MODEL_PATH=model/GOOG_prediction_model.keras
ENV SCALER_PATH=model/scaler.pkl
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Use exec form of CMD to ensure proper signal handling
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "app:app"]
