#!/bin/bash

# Configurar PYTHONPATH
export PYTHONPATH=/app:${PYTHONPATH}

# Instalar dependências de teste
pip install -r requirements-test.txt

# Criar diretório de dados de teste se não existir
mkdir -p data/test_chromadb

# Rodar testes com cobertura
pytest tests/ -v --cov=src/ --cov-report=term-missing
