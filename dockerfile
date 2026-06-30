# Estágio 1: Builder
FROM python:3.11-slim AS builder

# Instala uv
RUN pip install uv

# Define diretório de trabalho
WORKDIR /app

# Copia arquivos de dependência
COPY pyproject.toml uv.lock ./

# Instala dependências em um diretório específico
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv pip install --no-cache .
RUN uv sync --no-dev --frozen


# Estágio 2: Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copia apenas o ambiente virtual montado no estágio anterior
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copia o código fonte
COPY . .

# Define variáveis de ambiente
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Comando para rodar a pipeline ou o serviço
CMD ["python", "main.py"]