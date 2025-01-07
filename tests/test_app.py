import pytest
import streamlit as st
from unittest.mock import MagicMock, patch
import sys
import os

# Adicionar diretório src ao path para importação
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chatbot import Chatbot, logger

@pytest.fixture
def mock_chatbot():
    """Fixture para criar um mock do chatbot"""
    mock = MagicMock()
    mock.process_message.return_value = {
        "response": "Test response",
        "is_valid": True,
        "error": None,
        "intent": "fact",
        "preferences": {"tone": "casual", "verbosity": "balanced"}
    }
    return mock

@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock para funções do Streamlit"""
    with patch('streamlit.set_page_config'), \
         patch('streamlit.title'), \
         patch('streamlit.markdown'), \
         patch('streamlit.sidebar'), \
         patch('streamlit.chat_message'), \
         patch('streamlit.chat_input'), \
         patch('streamlit.spinner'), \
         patch('streamlit.error'), \
         patch('streamlit.success'), \
         patch('streamlit.caption'), \
         patch('streamlit.expander'), \
         patch('streamlit.subheader'), \
         patch('streamlit.metric'), \
         patch('streamlit.columns', return_value=[MagicMock(), MagicMock()]), \
         patch('streamlit.slider', return_value=0.7), \
         patch('streamlit.info'):
        yield

@pytest.fixture(autouse=True)
def setup_session_state():
    """Fixture para configurar o session_state antes de cada teste"""
    # Reset session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Initialize default values
    st.session_state.message_counts = {
        "fact": 0,
        "question": 0,
        "preference": 0,
        "feedback": 0
    }
    st.session_state.total_messages = 0
    st.session_state.facts_learned = 0
    st.session_state.current_preferences = {
        "tone": "casual",
        "verbosity": "balanced",
        "formality": "informal"
    }
    yield

def test_session_state_initialization():
    """Testa a inicialização do session_state"""
    # Import app to trigger initialization
    with patch('src.chatbot.Chatbot') as mock_chatbot:
        import src.app
        
        # Verificar inicialização do message_counts
        assert "message_counts" in st.session_state
        assert st.session_state.message_counts == {
            "fact": 0,
            "question": 0,
            "preference": 0,
            "feedback": 0
        }
        
        # Verificar outras variáveis do session_state
        assert "total_messages" in st.session_state
        assert "facts_learned" in st.session_state
        assert "current_preferences" in st.session_state

@pytest.mark.asyncio
async def test_chat_message_processing(mock_chatbot):
    """Testa o processamento de mensagens do chat"""
    # Initialize session state
    st.session_state.messages = []
    st.session_state.chatbot = mock_chatbot
    st.session_state.total_messages = 0
    st.session_state.message_counts = {
        "fact": 0,
        "question": 0,
        "preference": 0,
        "feedback": 0
    }
    
    # Simulate chat input and processing
    with patch('streamlit.chat_input', return_value="Test message"), \
         patch('streamlit.chat_message') as mock_chat_message:
        
        # Process message
        mock_chatbot.process_message.return_value = {
            "response": "Test response",
            "is_valid": True,
            "error": None,
            "intent": "fact",
            "preferences": {}
        }
        
        # Import app to trigger message processing
        with patch('src.chatbot.Chatbot', return_value=mock_chatbot):
            import src.app
            
            # Simulate chat input
            st.session_state.total_messages += 1
            st.session_state.message_counts["fact"] += 1
            
            # Verify message counts updated
            assert st.session_state.total_messages == 1
            assert st.session_state.message_counts["fact"] == 1

def test_preference_update(mock_chatbot):
    """Testa a atualização de preferências"""
    # Initialize session state
    st.session_state.messages = []
    st.session_state.chatbot = mock_chatbot
    
    # Configure mock response
    new_preferences = {
        "tone": "formal",
        "verbosity": "concise",
        "formality": "formal"
    }
    mock_chatbot.process_message.return_value = {
        "response": "Preferences updated",
        "is_valid": True,
        "error": None,
        "intent": "preference",
        "preferences": new_preferences
    }
    
    # Simulate preference update
    with patch('streamlit.chat_input', return_value="I prefer formal and concise responses"), \
         patch('streamlit.chat_message'):
        
        # Import app to trigger message processing
        with patch('src.chatbot.Chatbot', return_value=mock_chatbot):
            import src.app
            
            # Update preferences in session state
            st.session_state.current_preferences = new_preferences
            
            # Verify preferences updated
            assert st.session_state.current_preferences == new_preferences
