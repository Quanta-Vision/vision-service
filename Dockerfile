# Use official Python slim image
FROM python:3.12-slim

# Set env variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Ho_Chi_Minh

# Set working directory
WORKDIR /app

# System dependencies for ML + OpenCV + Ultralytics/InsightFace
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
        libgl1 \
        && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY ./app /app/app

# Optional: copy .env file if using for development (remove in production)
# COPY .env .env

# Expose FastAPI default port
EXPOSE 8001

# Command to run the app (edit path if your entrypoint is elsewhere)
CMD ["python", "-m", "app.main"]
