FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[service]"

# Run as a non-root user — defense in depth on top of Cloud Run's own
# container sandboxing, not a substitute for it.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8080

CMD ["llm-gateway"]
