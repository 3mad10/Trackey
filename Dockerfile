FROM python:3.11-slim

# Install system dependencies for OpenCV and Git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project definition
COPY pyproject.toml .
COPY trackey/ trackey/
COPY scripts/ scripts/

# Build and install dependencies:
# 1. Install numpy first to satisfy torchreid's setup.py requirements
RUN pip install --no-cache-dir "numpy==1.26.4"

# 2. Install trackey with all optional dependencies
RUN pip install --no-cache-dir -e .[all]

# 3. Install torchreid with --no-build-isolation
RUN pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/KaiyangZhou/deep-person-reid.git@f8cd150fdf77e8d9e1ed143b7f308c2c609ded50"

# Copy configuration and model weights if present
COPY base_pipeline.yaml .

EXPOSE 8000

CMD ["trackey", "run", "base_pipeline.yaml"]
