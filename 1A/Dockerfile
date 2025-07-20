# Use an official Python runtime as a parent image, specifying the AMD64 platform
FROM --platform=linux/amd64 python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and to run in unbuffered mode
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by some Python packages (e.g., PyMuPDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application script, model, and other necessary assets
COPY run.py .
COPY multilingual_heading_model_v2.h5 .
COPY label_encoder_v2.pkl .
COPY scaler_v2.pkl .
COPY numerical_features_v2.json .

# Create the input and output directories as specified in the challenge document
RUN mkdir -p /app/input /app/output

# Specify the command to run on container startup
CMD ["python", "run.py"]
