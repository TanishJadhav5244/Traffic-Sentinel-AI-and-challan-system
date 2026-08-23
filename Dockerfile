# Multi-stage Dockerfile for Traffic Sentinel AI
FROM python:3.10-slim as base

# Install system dependencies for OpenCV, Tesseract OCR, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY . .

# Create output directories
RUN mkdir -p violations/crops violations/challans models

# Expose ports: 8501 for Streamlit Dashboard, 8000 for FastAPI REST API
EXPOSE 8501 8000

# Default command runs both Streamlit UI and FastAPI backend
CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
