FROM python:3.12-slim-bookworm

WORKDIR /app

# LightGBM OpenMP dependency
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Non-root user and writable directories
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home app \
    && mkdir -p /app/runtime /app/output /app/data/co2 \
    && chown -R app:app /app

# Application source code
COPY --chown=app:app main.py .
COPY --chown=app:app analysis/ ./analysis/
COPY --chown=app:app data/ ./data/
COPY --chown=app:app models/ ./models/
COPY --chown=app:app profiling_data/ ./profiling_data/
COPY --chown=app:app visualization/ ./visualization/

USER app

CMD ["python", "main.py"]