FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install both build-time and runtime system dependencies required by
# Pillow, librosa/moviepy (ffmpeg), soundfile and OpenCV. We remove the
# build tools after pip install to keep the final image leaner.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc pkg-config git ca-certificates \
    libjpeg-dev zlib1g-dev libtiff5-dev libfreetype6-dev liblcms2-dev libwebp-dev libopenjp2-7-dev \
    ffmpeg libgl1 libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Upgrade packaging tools and prefer binary wheels where available to avoid
# building packages from source (this reduces build time and errors).
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --prefer-binary --no-warn-script-location -r requirements.txt \
    # purge build tools used only for compiling wheels to reduce final size
    && apt-get purge -y --auto-remove build-essential gcc pkg-config git \
    && rm -rf /var/lib/apt/lists/* /tmp/*

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD ["waitress-serve", "--listen=0.0.0.0:8080", "main:app"]
