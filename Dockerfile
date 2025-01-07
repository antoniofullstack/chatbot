# Estágio base
FROM python:3.11-slim as base

WORKDIR /app

# Configurar PYTHONPATH
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Estágio de teste
FROM base as test
COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

COPY . .

# Estágio de produção
FROM base as prod
CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]