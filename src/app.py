import streamlit as st
from dotenv import load_dotenv
from src.chatbot import Chatbot, logger

# Load environment variables
load_dotenv()

# Configurar página Streamlit
st.set_page_config(
    page_title="Chatbot de Aprendizado",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'Chatbot de Aprendizado Contínuo'
    }
)

# Inicializar contador de tipos de mensagens no session_state
if "message_counts" not in st.session_state:
    st.session_state.message_counts = {
        "fact": 0,
        "question": 0,
        "preference": 0,
        "feedback": 0
    }

# Sidebar com configurações e estatísticas
with st.sidebar:
    st.title("⚙️ Configurações")
    
    # Configurações do modelo
    st.subheader("Configurações do Modelo")
    st.session_state.temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Valores mais altos tornam a saída mais aleatória, valores mais baixos a tornam mais determinística"
    )
    
    # Estatísticas
    st.subheader("📊 Estatísticas")
    if "total_messages" not in st.session_state:
        st.session_state.total_messages = 0
    if "facts_learned" not in st.session_state:
        st.session_state.facts_learned = 0
        
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de Mensagens", st.session_state.total_messages)
        st.metric("Fatos Aprendidos", st.session_state.facts_learned)
    with col2:
        st.metric("Perguntas Feitas", st.session_state.message_counts["question"])
        st.metric("Preferências Definidas", st.session_state.message_counts["preference"])
    
    # Preferências Atuais
    st.subheader("🎯 Preferências Atuais")
    if "current_preferences" not in st.session_state:
        st.session_state.current_preferences = {
            "tone": "casual",
            "verbosity": "balanced",
            "formality": "informal"
        }
    
    for pref, value in st.session_state.current_preferences.items():
        st.info(f"{pref.title()}: {value}")
    
    # Sobre
    st.subheader("ℹ️ Sobre")
    st.markdown("""
    Este chatbot aprende com conversas e armazena fatos validados.
    - Usa Groq LLM para processamento
    - ChromaDB para armazenamento de conhecimento
    - Construído com LangChain & LangGraph
    
    ### Dicas:
    - Compartilhe fatos sobre qualquer tópico
    - Faça perguntas sobre informações armazenadas
    - Defina suas preferências (ex: "Prefiro respostas formais")
    - Forneça feedback para me ajudar a melhorar
    """)

# Interface principal do chat
st.title("🤖 Chatbot de Aprendizado")
st.markdown("""
Bem-vindo! Sou um chatbot que pode aprender com nossas conversas.
- Compartilhe fatos comigo e eu vou validá-los e lembrá-los
- Pergunte-me sobre o que aprendi
- Me diga suas preferências
- Forneça feedback para me ajudar a melhorar
""")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.info("Initializing chatbot...")
    st.session_state.chatbot = Chatbot()
    logger.info("Chatbot initialized successfully")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Mostrar badges para diferentes tipos de mensagens
        if message["role"] == "user" and message.get("intent"):
            st.caption(f"Tipo: {message['intent']}")
        
        # Mostrar status de validação para fatos
        if message.get("is_valid") is True:
            st.success("✅ Esta informação foi validada e armazenada")
        elif message.get("is_valid") is False:
            st.error("❌ Esta informação não pôde ser validada")
        
        # Mostrar preferências atualizadas
        if message.get("preferences") and message["role"] == "assistant":
            with st.expander("Preferências Atualizadas"):
                for pref, value in message["preferences"].items():
                    st.info(f"{pref.title()}: {value}")

# Chat input
if prompt := st.chat_input("O que você gostaria de discutir?"):
    # Increment message counter
    st.session_state.total_messages += 1
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process the message
    try:
        # Atualizar temperatura do modelo
        st.session_state.chatbot.llm.temperature = st.session_state.temperature
        
        # Processar mensagem
        response = st.session_state.chatbot.process_message(prompt)
        
        # Atualizar contadores
        if response.get("intent"):
            intent = response["intent"]
            if intent in st.session_state.message_counts:
                st.session_state.message_counts[intent] += 1
            
            # Atualizar contador de fatos aprendidos
            if intent == "fact" and response.get("is_valid", False):
                st.session_state.facts_learned += 1
        
        # Check if there was an error
        if response.get("error"):
            st.error(response["response"])
        else:
            # Display response
            st.markdown(response["response"])
            
            # Add response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["response"],
                "is_valid": response["is_valid"],
                "intent": response["intent"],
                "preferences": response["preferences"]
            })
            
    except Exception as e:
        error_msg = f"Erro ao processar mensagem: {e}"
        logger.error(error_msg)
        st.error(error_msg)
