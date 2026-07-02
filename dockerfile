# Estágio 1: Builder
FROM python:3.11-slim AS builder

RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv pip install --no-cache .
RUN uv sync --no-dev --frozen

# Estágio 2: Runtime
FROM python:3.11-slim AS runtime


RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Atualize também a variável de ambiente para refletir a nova versão
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]