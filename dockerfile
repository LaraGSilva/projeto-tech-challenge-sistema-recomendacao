# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.10-slim AS builder

# Instalar o uv diretamente da imagem oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Configurar diretório de trabalho
WORKDIR /app

# Copiar apenas os arquivos de dependência primeiro para aproveitar o cache do Docker
COPY pyproject.toml uv.lock ./

# Criar um ambiente virtual e instalar as dependências
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Sincronizar as dependências garantindo que o lockfile seja respeitado
RUN uv sync --frozen --no-dev

# ==========================================
# Stage 2: Runtime
# ==========================================
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copiar o ambiente virtual do stage builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copiar o restante do código da aplicação
COPY . .

# O comando padrão pode ser ajustado para iniciar uma API FastAPI ou rodar um script de treino
CMD ["python", "src/recommender/models/train_baselines.py"]