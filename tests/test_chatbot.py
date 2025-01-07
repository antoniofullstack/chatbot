import pytest
from src.chatbot import Chatbot, ChatState
import os
import shutil
from unittest.mock import MagicMock, patch
import json

# Fixture para criar e limpar o ambiente de teste
@pytest.fixture
def test_chatbot():
    """Fixture para criar uma instância do chatbot com mocks"""
    # Criar mocks
    mock_llm = MagicMock()
    mock_chroma = MagicMock()
    mock_graph = MagicMock()
    mock_workflow = MagicMock()
    
    # Configurar comportamento padrão dos mocks
    mock_llm.return_value.content = ""
    mock_chroma.similarity_search.return_value = [
        MagicMock(
            page_content="Test content",
            metadata={"type": "fact", "preferences": "{}"}
        )
    ]
    mock_workflow.invoke.return_value = {
        "input": "",
        "intent": "",
        "is_valid": True,
        "response": "Test response",
        "error": None,
        "preferences": {},
        "context": []
    }
    
    # Criar instância do chatbot com mocks
    with patch('src.chatbot.ChatGroq', return_value=mock_llm), \
         patch('src.chatbot.Chroma', return_value=mock_chroma), \
         patch('src.chatbot.HuggingFaceEmbeddings'), \
         patch('src.chatbot.StateGraph', return_value=mock_graph):
        chatbot = Chatbot()
        chatbot.llm = mock_llm
        chatbot.vector_store = mock_chroma
        chatbot.workflow = mock_workflow
        yield chatbot

def test_chatbot_initialization_error():
    """Testa erro na inicialização do chatbot"""
    with pytest.raises(Exception), \
         patch('src.chatbot.ChatGroq', side_effect=Exception("API Error")), \
         patch('src.chatbot.HuggingFaceEmbeddings'), \
         patch('src.chatbot.Chroma'), \
         patch('src.chatbot.StateGraph'):
        Chatbot()

def test_process_fact(test_chatbot):
    """Testa o processamento de um fato"""
    # Configurar mock do LLM para retornar "fact" como intent
    test_chatbot.llm.return_value.content = "fact"
    
    state = ChatState(
        input="The capital of France is Paris",
        intent="",
        is_valid=False,
        response="",
        error=None,
        preferences={},
        context=[]
    )
    
    # Testar processamento de input
    result = test_chatbot.process_input(state)
    assert result["intent"] == "fact"
    assert not result["error"]
    
    # Configurar mock para validação
    test_chatbot.llm.return_value.content = "true"
    
    # Testar validação do fato
    result = test_chatbot.validate_fact(result)
    assert result["is_valid"]
    assert not result["error"]
    
    # Testar armazenamento
    result = test_chatbot.store_information(result)
    assert not result["error"]
    test_chatbot.vector_store.add_documents.assert_called_once()

def test_process_preference(test_chatbot):
    """Testa o processamento de uma preferência"""
    # Configurar mock do LLM para retornar "preference" como intent
    test_chatbot.llm.return_value.content = "preference"
    
    state = ChatState(
        input="I prefer formal and concise responses",
        intent="",
        is_valid=False,
        response="",
        error=None,
        preferences={},
        context=[]
    )
    
    # Testar processamento de input
    result = test_chatbot.process_input(state)
    assert result["intent"] == "preference"
    assert not result["error"]
    
    # Configurar mock para extrair preferências
    test_chatbot.llm.return_value.content = "tone:formal\nverbosity:concise"
    
    # Testar atualização de preferências
    result = test_chatbot.update_preferences(result)
    assert result["preferences"]["tone"] == "formal"
    assert result["preferences"]["verbosity"] == "concise"
    assert not result["error"]
    
    # Configurar mock para armazenamento
    test_chatbot.vector_store.add_documents.reset_mock()
    test_chatbot.vector_store.add_documents.return_value = None
    
    # Definir is_valid como True para permitir o armazenamento
    result["is_valid"] = True
    
    # Testar armazenamento de preferências
    result = test_chatbot.store_information(result)
    assert not result["error"]
    test_chatbot.vector_store.add_documents.assert_called_once()

def test_context_retrieval(test_chatbot):
    """Testa a recuperação de contexto"""
    # Configurar mock do ChromaDB
    test_docs = [
        MagicMock(
            page_content="Paris is the capital of France",
            metadata={
                "type": "fact",
                "preferences": json.dumps({"tone": "formal"})
            }
        ),
        MagicMock(
            page_content="The Eiffel Tower is in Paris",
            metadata={
                "type": "fact",
                "preferences": None
            }
        )
    ]
    test_chatbot.vector_store.similarity_search.return_value = test_docs
    
    query_state = ChatState(
        input="Tell me about Paris",
        intent="question",
        is_valid=True,
        response="",
        error=None,
        preferences={},
        context=[]
    )
    
    result = test_chatbot.get_context(query_state)
    assert len(result["context"]) == 2
    assert "Paris" in result["context"][0]["content"]
    assert not result["error"]
    test_chatbot.vector_store.similarity_search.assert_called_once()

def test_invalid_fact_rejection(test_chatbot):
    """Testa a rejeição de fatos inválidos"""
    # Configurar mock do LLM para retornar "fact" como intent
    test_chatbot.llm.return_value.content = "fact"
    
    state = ChatState(
        input="The moon is made of cheese",
        intent="",
        is_valid=False,
        response="",
        error=None,
        preferences={},
        context=[]
    )
    
    # Testar processamento de input
    result = test_chatbot.process_input(state)
    assert result["intent"] == "fact"
    
    # Configurar mock para validação falsa
    test_chatbot.llm.return_value.content = "false"
    
    # Testar validação do fato
    result = test_chatbot.validate_fact(result)
    assert not result["is_valid"]
    assert not result["error"]
    
    # Testar que fatos inválidos não são armazenados
    result = test_chatbot.store_information(result)
    assert not result["error"]
    test_chatbot.vector_store.add_documents.assert_not_called()

def test_preference_adaptation(test_chatbot):
    """Testa a adaptação às preferências do usuário"""
    # Configurar mock do LLM para retornar um JSON válido
    test_chatbot.llm.return_value.content = '{"tom": "formal", "verbosidade": "detalhada"}'
    
    # 1. Definir preferências
    pref_state = ChatState(
        input="I prefer formal and detailed responses",
        intent="preference",
        is_valid=True,
        response="",
        error=None,
        preferences={},
        context=[]
    )
    
    # Atualizar preferências
    result = test_chatbot.update_preferences(pref_state)
    assert result["preferences"]["tom"] == "formal"
    assert result["preferences"]["verbosidade"] == "detalhada"
    
    # 2. Testar resposta com preferências
    test_chatbot.llm.return_value.content = "A formal and detailed response"
    
    response = test_chatbot.generate_response(result)
    assert not response["error"]
    assert "formal and detailed response" in response["response"].lower()

def test_error_handling(test_chatbot):
    """Testa o tratamento de erros"""
    # 1. Erro no LLM
    test_chatbot.llm.side_effect = Exception("LLM Error")
    test_chatbot.workflow.invoke.return_value = {
        "input": "",
        "intent": "",
        "is_valid": False,
        "response": "",
        "error": "LLM Error",
        "preferences": {},
        "context": []
    }
    
    result = test_chatbot.process_message("")
    assert result["error"]
    assert not result["is_valid"]
    assert "LLM Error" in result["response"]
    
    # 2. Erro no ChromaDB
    test_chatbot.llm.side_effect = None
    test_chatbot.llm.return_value.content = "fact"
    test_chatbot.vector_store.similarity_search.side_effect = Exception("DB Error")
    
    state = ChatState(
        input="Test fact",
        intent="fact",
        is_valid=False,
        response="",
        error=None,
        preferences={},
        context=[]
    )
    
    result = test_chatbot.get_context(state)
    assert result["error"]
    assert "DB Error" in result["error"]

def test_complete_conversation_flow(test_chatbot):
    """Testa um fluxo completo de conversa"""
    # Configurar mocks para diferentes etapas
    def mock_llm_responses(*args, **kwargs):
        prompt = args[0][0].content if args and args[0] else ""
        if "prefer" in prompt.lower():
            return MagicMock(content="preference")
        elif "Earth" in prompt:
            return MagicMock(content="fact")
        else:
            return MagicMock(content="question")
    
    test_chatbot.llm.side_effect = mock_llm_responses
    
    # 1. Definir preferências
    test_chatbot.workflow.invoke.return_value = {
        "input": "I prefer formal and concise responses",
        "intent": "preference",
        "is_valid": True,
        "response": "Preferences updated",
        "error": None,
        "preferences": {"tom": "formal", "verbosidade": "concisa"},
        "context": []
    }
    response = test_chatbot.process_message("I prefer formal and concise responses")
    assert not response["error"]
    
    # 2. Compartilhar um fato
    test_chatbot.workflow.invoke.return_value = {
        "input": "The Earth orbits around the Sun",
        "intent": "fact",
        "is_valid": True,
        "response": "Fact validated and stored",
        "error": None,
        "preferences": {"tom": "formal", "verbosidade": "concisa"},
        "context": []
    }
    response = test_chatbot.process_message("The Earth orbits around the Sun")
    assert response["is_valid"]
    assert not response["error"]
    
    # 3. Fazer uma pergunta
    test_chatbot.workflow.invoke.return_value = {
        "input": "What can you tell me about the Earth?",
        "intent": "question",
        "is_valid": True,
        "response": "The Earth orbits around the Sun",
        "error": None,
        "preferences": {"tom": "formal", "verbosidade": "concisa"},
        "context": [
            {
                "content": "The Earth orbits around the Sun",
                "metadata": {"type": "fact"}
            }
        ]
    }
    response = test_chatbot.process_message("What can you tell me about the Earth?")
    assert not response["error"]

def test_graph_setup(test_chatbot):
    """Testa a configuração do grafo de conversa"""
    with patch('src.chatbot.ChatGroq'), \
         patch('src.chatbot.HuggingFaceEmbeddings'), \
         patch('src.chatbot.Chroma'), \
         patch('src.chatbot.StateGraph') as mock_graph_class:
        
        # Configurar o mock do grafo
        mock_graph = MagicMock()
        mock_graph_class.return_value = mock_graph
        mock_graph.add_node.return_value = mock_graph
        mock_graph.add_edge.return_value = mock_graph
        mock_graph.set_entry_point.return_value = mock_graph
        mock_graph.compile.return_value = mock_graph
        
        # Criar chatbot para testar configuração do grafo
        chatbot = Chatbot()
        
        # Verificar se os nós foram adicionados
        mock_graph.add_node.assert_any_call("process_input", chatbot.process_input)
        mock_graph.add_node.assert_any_call("validate_fact", chatbot.validate_fact)
        mock_graph.add_node.assert_any_call("update_preferences", chatbot.update_preferences)
        
        # Verificar se as arestas foram configuradas
        mock_graph.add_edge.assert_called()
        mock_graph.set_entry_point.assert_called_with("process_input")
        mock_graph.compile.assert_called_once()

def test_error_in_graph_setup():
    """Testa erro na configuração do grafo"""
    with pytest.raises(Exception), \
         patch('src.chatbot.ChatGroq'), \
         patch('src.chatbot.HuggingFaceEmbeddings'), \
         patch('src.chatbot.Chroma'), \
         patch('src.chatbot.StateGraph', side_effect=Exception("Graph Error")):
        Chatbot()

def test_store_information_with_error(test_chatbot):
    """Testa armazenamento com erro prévio"""
    state = ChatState(
        input="Test fact",
        intent="fact",
        is_valid=True,
        response="",
        error="Previous error",
        preferences={},
        context=[]
    )
    
    result = test_chatbot.store_information(state)
    assert result["error"] == "Previous error"
    test_chatbot.vector_store.add_documents.assert_not_called()

def test_generate_response_with_error(test_chatbot):
    """Testa geração de resposta com erro"""
    test_chatbot.llm.side_effect = Exception("Response Error")
    
    state = ChatState(
        input="Test input",
        intent="question",
        is_valid=True,
        response="",
        error=None,
        preferences={},
        context=[]
    )
    
    result = test_chatbot.generate_response(state)
    assert result["error"] == "Response Error"
    assert "I apologize" in result["response"]

def test_get_context_with_error(test_chatbot):
    """Testa o comportamento de get_context quando há erro"""
    state = {
        "input": "test input",
        "intent": "",
        "is_valid": False,
        "response": "",
        "error": "Previous error",
        "preferences": {},
        "context": []
    }
    
    result = test_chatbot.get_context(state)
    assert result["error"] == "Previous error"

def test_validate_fact_with_context(test_chatbot):
    """Testa a validação de fatos com contexto existente"""
    # Configurar mock do LLM para retornar "true"
    test_chatbot.llm.return_value.content = "true"
        
    state = ChatState(
        input="The Earth is round",
        intent="fact",
        is_valid=False,
        response="",
        error=None,
        preferences={},
        context=[
            {
                "content": "The Earth is a sphere",
                "metadata": {"type": "fact"}
            }
        ]
    )
    
    result = test_chatbot.validate_fact(state)
    assert result["is_valid"] == True

def test_update_preferences_with_multiple_values(test_chatbot):
    """Testa a atualização de preferências com múltiplos valores"""
    # Configurar mock do LLM para retornar um JSON válido
    test_chatbot.llm.return_value.content = '{"tom": "formal", "verbosidade": "concisa"}'
        
    state = ChatState(
        input="I prefer formal and concise responses",
        intent="preference",
        is_valid=False,
        response="",
        error=None,
        preferences={},
        context=[]
    )
    
    result = test_chatbot.update_preferences(state)
    assert result["preferences"]["tom"] == "formal"
    assert result["preferences"]["verbosidade"] == "concisa"

def test_store_information_with_preferences(test_chatbot):
    """Testa o armazenamento de informações com preferências"""
    state = ChatState(
        input="I prefer formal responses",
        intent="preference",
        is_valid=True,
        response="",
        error=None,
        preferences={"tom": "formal"},
        context=[]
    )
    
    result = test_chatbot.store_information(state)
    test_chatbot.vector_store.add_documents.assert_called_once()
    assert not result["error"]

def test_generate_response_with_context(test_chatbot):
    """Testa a geração de resposta com contexto"""
    # Configurar mock do LLM para retornar uma resposta específica
    test_chatbot.llm.return_value.content = "The Earth is round based on scientific evidence"
        
    state = ChatState(
        input="What do you know about Earth?",
        intent="question",
        is_valid=True,
        response="",
        error=None,
        preferences={"tom": "casual"},
        context=[
            {
                "content": "The Earth is round",
                "metadata": {"type": "fact"}
            }
        ]
    )
    
    result = test_chatbot.generate_response(state)
    assert result["response"] == "The Earth is round based on scientific evidence"

def test_process_message_with_workflow_error(test_chatbot):
    """Testa o processamento de mensagem quando o workflow falha"""
    test_chatbot.workflow.invoke.side_effect = Exception("Workflow error")
    
    result = test_chatbot.process_message("test message")
    assert result["error"]
    assert not result["is_valid"]
    assert "error" in result["response"].lower()