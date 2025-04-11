FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model and application code
COPY model/ ./model/
COPY app.py .

# Verify model files exist
RUN ls -la model/

# Set environment variables
ENV MODEL_PATH=model/GOOG_prediction_model.keras
ENV SCALER_PATH=model/scaler.pkl
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Run the application
CMD ["python", "-u", "app.py"]
