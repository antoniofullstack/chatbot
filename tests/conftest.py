import pytest
from unittest.mock import MagicMock
import os
import sys

# Adicionar diretório raiz ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock das configurações
@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    mock_settings = MagicMock()
    mock_settings.GROQ_API_KEY = "test-key"
    mock_settings.MODEL_NAME = "test-model"
    mock_settings.TEMPERATURE = 0.7
    mock_settings.EMBEDDING_MODEL = "test-embeddings"
    
    # Aplicar o mock
    monkeypatch.setattr('src.config.settings', mock_settings)
    return mock_settings
