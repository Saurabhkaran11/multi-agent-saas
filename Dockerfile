# Use an explicit, lightweight Python runtime
FROM python:3.11-slim

# Set system environment adjustments
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker's caching mechanisms
COPY requirements.txt .

# Install dependencies directly into the container system layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose port 8000 to the public internet router
EXPOSE 8000

# Launch Uvicorn and force it to listen on 0.0.0.0 so Zeabur can route traffic
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
