FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

# COPY . .
EXPOSE 7000

#CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m",]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000", "--reload"]
# CMD ["tail", "-f", "/dev/null"]

# Use a lightweight base image
# FROM python:3.10-slim as base

# # Set the working directory
# WORKDIR /app

# # Install system dependencies (if needed)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# # Install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the application code
# COPY . .

# # Run the FastAPI server
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000"]
