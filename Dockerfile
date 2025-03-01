# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY database/ ./database/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory for models
RUN mkdir -p models

# Set environment variables
ENV PYTHONPATH=/app
ENV TESTING=0
ENV USE_MOCK_DATA=0

# Make port 8000 available
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.utils:app", "--host", "0.0.0.0", "--port", "8000"] 