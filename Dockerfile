# Use a lightweight Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for building some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variable to force CPU usage (optional, but good for clarity)
ENV CUDA_VISIBLE_DEVICES=""

# Default command: Run the evaluation script to show proof of work
CMD ["python", "src/evaluation/evaluate.py"]
