# Fetcher Container for AWS HealthImaging MLOps Pipeline
# Retrieves pixel data from HealthImaging and prepares it for ML training

FROM python:3.10-slim

# Set working directory
WORKDIR /opt/ml/code

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the fetcher script
COPY fetch.py .

# Set the entrypoint
ENTRYPOINT ["python", "fetch.py"]
