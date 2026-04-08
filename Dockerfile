### Builder stage: build wheels (including CPU-only PyTorch wheels)
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /wheels

# Build-time deps for compiling wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc pkg-config ca-certificates \
    libjpeg-dev zlib1g-dev libtiff5-dev libfreetype6-dev liblcms2-dev libwebp-dev libopenjp2-7-dev \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Upgrade pip tooling then build wheels. Use PyTorch CPU wheel index so the
# CPU-only wheels are fetched (avoids pulling CUDA-enabled artifacts).
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip wheel --wheel-dir=/wheels -r requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html

### Runtime stage: minimal runtime with only necessary system libs
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Runtime libraries required by Pillow/ffmpeg/soundfile/OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libsndfile1 ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy prebuilt wheels and application code
COPY --from=builder /wheels /wheels
COPY . .

# Install from wheels (no network). Clean up caches to reduce image size.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels /root/.cache/pip

ENV PORT=8080
EXPOSE 8080

CMD ["waitress-serve", "--listen=0.0.0.0:8080", "main:app"]
