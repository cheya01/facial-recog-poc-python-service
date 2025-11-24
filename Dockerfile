# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and face detection
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fv.py .
COPY .env .

# Create directory for temporary files
RUN mkdir -p /tmp/face-embed

# Expose port
EXPOSE 5010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5010/health')"

# Run the application with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5010", "--workers", "2", "--threads", "4", "--timeout", "120", "--worker-class", "sync", "--access-logfile", "-", "--error-logfile", "-", "fv:app"]
