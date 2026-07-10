############################
# Stage: Builder
############################
FROM python:3.11-slim AS builder

WORKDIR /app

# Instala o uv
RUN pip install --no-cache-dir uv

# Faz o uv criar o ambiente virtual fora de /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# Copia apenas os arquivos de dependências
COPY pyproject.toml uv.lock ./

# Instala as dependências
RUN uv sync --frozen

# Copia o restante do projeto
COPY . .

############################
# Stage: Runtime
############################
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app

# Copia o ambiente virtual e a aplicação
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
