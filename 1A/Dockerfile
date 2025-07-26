# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the trained model files into the container
COPY run1.py .
COPY english_distilbert_model.h5 .
COPY english_label_encoder.pkl .
COPY english_scaler.pkl .
COPY english_numerical_features.json .

# Expose port 5000 to the outside world
EXPOSE 5000

# Set the default command to run the Flask API
CMD ["python", "run1.py"]