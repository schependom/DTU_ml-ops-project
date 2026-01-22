FROM python:3.12-slim

WORKDIR /app

# Install dependencies directly to minimize complexity
RUN pip install fastapi uvicorn python-dotenv torch transformers wandb google-cloud-storage pytorch-lightning hydra-core torchmetrics pydantic pandas prometheus-client --no-cache-dir

# Copy source code and configs
COPY src/ /app/src/
COPY configs/ /app/configs/
# Copy pyproject.toml just in case (though we use pip)
COPY pyproject.toml /app/

# Set PYTHONPATH so that 'ml_ops_project' can be imported
ENV PYTHONPATH=/app/src

# Default port for Cloud Run
ENV PORT=8080

EXPOSE $PORT

# Use exec form to handle signals correctly
CMD ["sh", "-c", "uvicorn ml_ops_project.api:app --port $PORT --host 0.0.0.0 --workers 1"]
