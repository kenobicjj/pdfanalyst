# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose the port Flask runs on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"] 