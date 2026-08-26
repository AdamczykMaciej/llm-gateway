FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[service]"

EXPOSE 8080

CMD ["llm-gateway"]
