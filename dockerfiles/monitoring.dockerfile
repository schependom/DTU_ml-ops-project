FROM python:3.12-slim

WORKDIR /app

# Install dependencies directly to minimize complexity
RUN pip install fastapi nltk evidently google-cloud-storage pandas datasets

# Copy source code and configs
COPY src/ /app/src/
COPY configs/ /app/configs/

# Set PYTHONPATH so that 'ml_ops_project' can be imported
ENV PYTHONPATH=/app/src

# Default port for Cloud Run
ENV PORT=8080

EXPOSE $PORT

# Use exec form to handle signals correctly
CMD ["sh", "-c", "uvicorn ml_ops_project.monitoring:app --port $PORT --host 0.0.0.0 --workers 1"]