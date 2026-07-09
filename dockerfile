# # Estágio 1: Builder
# FROM python:3.11-slim AS builder

# RUN pip install uv
# WORKDIR /app
# COPY pyproject.toml uv.lock ./

# RUN uv venv /opt/venv
# ENV VIRTUAL_ENV=/opt/venv
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# RUN uv pip install --no-cache .
# RUN uv sync --no-dev --frozen

# # Estágio 2: Runtime
# FROM python:3.11-slim AS runtime


# RUN apt-get update && apt-get install -y --no-install-recommends \
#     openjdk-21-jre-headless \
#     && rm -rf /var/lib/apt/lists/*



# WORKDIR /app

# COPY --from=builder /opt/venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# COPY . .

# ENV PYTHONPATH=/app
# ENV PYTHONUNBUFFERED=1

# CMD ["python", "main.py"]


############################
# Stage: Builder
############################
FROM python:3.11-slim AS builder

# Otimizações para Python e UV
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Instala apenas o UV
RUN pip install --no-cache-dir uv

# Copia apenas os arquivos de definição de dependências
COPY pyproject.toml uv.lock ./

# Instala SOMENTE as dependências do projeto (ignora dev)
# O --frozen garante que ele instale exatamente o que está no lock
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copia o código fonte para dentro do builder
COPY . .

############################
# Stage: Runtime
############################
FROM python:3.11-slim AS runtime

# Variáveis para rodar no ambiente virtual criado
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

WORKDIR /app

# Copia apenas o ambiente virtual montado e o código necessário
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Removemos qualquer instalação de APT desnecessária
# O sistema já tem o essencial para rodar o Python (o slim já vem pronto)

CMD ["python", "main.py"]