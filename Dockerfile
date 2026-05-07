# ── base stage ───────────────────────────────────────────────
FROM python:3.11-slim AS base

# prevent .pyc files — not needed in containers
ENV PYTHONDONTWRITEBYTECODE=1

# ensure stdout/stderr flush immediately — critical for container logs
ENV PYTHONUNBUFFERED=1

# MLflow tracking URI — uses local filesystem by default
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

WORKDIR /app

# ── dependencies stage ────────────────────────────────────────
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── final stage ───────────────────────────────────────────────
FROM dependencies AS final

# copy source code
COPY src/     ./src/
COPY data/    ./data/
COPY monitoring/ ./monitoring/
COPY tests/   ./tests/

# create directories for runtime artifacts
RUN mkdir -p artifacts monitoring/reports

# create a non-root user for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# expose ports:
#   8000 = FastAPI inference server
#   5000 = MLflow tracking UI
EXPOSE 8000 5000

# default: run training
# override at runtime:
#   docker run mlops-pipeline python src/evaluate.py
#   docker run -p 8000:8000 mlops-pipeline python src/predict.py
CMD ["python", "src/train.py"]