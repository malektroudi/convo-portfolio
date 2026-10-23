# Cloud Run runs this. Keep it boring: a wheel install, a non-root user, $PORT.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    PORT=8080

WORKDIR /app

# Dependencies first, so a source-only change does not reinstall the world.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 convo
USER convo

EXPOSE 8080

# Cloud Run sends SIGTERM; python -m handles it and Gradio shuts the queue down.
CMD ["python", "-m", "convo.hub"]
