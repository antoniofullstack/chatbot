import os
from typing import Dict, List, TypedDict
import logging
import json
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from src.config import settings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatState(TypedDict):
    input: str
    intent: str
    is_valid: bool
    response: str
    error: str | None
    preferences: Dict[str, str]
    context: List[Dict]

class Chatbot:
    def __init__(self):
        try:
            logger.info("Inicializando Chatbot...")
            self.llm = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model_name=settings.MODEL_NAME,
                temperature=settings.TEMPERATURE,
                max_retries=3,
                request_timeout=30
            )
            logger.info("LLM inicializado com sucesso")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Modelo de embeddings inicializado com sucesso")
            
            self.vector_store = Chroma(
                persist_directory="data/chromadb",
                embedding_function=self.embeddings
            )
            logger.info("Vector store inicializado com sucesso")
            
            self.setup_graph()
            logger.info("Configuração do grafo completada")
            
            # Inicializar preferências padrão
            self.default_preferences = {
                "tom": "casual",
                "verbosidade": "balanceada",
                "formalidade": "informal"
            }
            
        except Exception as e:
            logger.error(f"Erro ao inicializar Chatbot: {e}")
            raise

    def setup_graph(self):
        try:
            # Define the conversation flow graph
            self.graph = StateGraph(state_schema=ChatState)

            # Add nodes
            self.graph.add_node("process_input", self.process_input)
            self.graph.add_node("get_context", self.get_context)
            self.graph.add_node("validate_fact", self.validate_fact)
            self.graph.add_node("update_preferences", self.update_preferences)
            self.graph.add_node("store_information", self.store_information)
            self.graph.add_node("generate_response", self.generate_response)

            # Define the edges
            self.graph.add_edge('process_input', 'get_context')
            self.graph.add_edge('get_context', 'validate_fact')
            self.graph.add_edge('validate_fact', 'update_preferences')
            self.graph.add_edge('update_preferences', 'store_information')
            self.graph.add_edge('store_information', 'generate_response')
            self.graph.add_edge('generate_response', END)

            # Set the entry point
            self.graph.set_entry_point("process_input")

            # Compile the graph
            self.workflow = self.graph.compile()
        except Exception as e:
            logger.error(f"Erro ao configurar o grafo: {e}")
            raise

    def process_input(self, state: ChatState) -> ChatState:
        """Processa a entrada do usuário e determina a intenção"""
        try:
            logger.info(f"Processando entrada: {state['input']}")
            
            system_content = """Você é um classificador de intenções.
            
            IMPORTANTE: Responda APENAS com UMA das seguintes palavras, sem pontuação ou texto adicional:
            - fact (quando o usuário compartilha uma informação factual)
            - question (quando o usuário faz uma pergunta)
            - preference (quando o usuário expressa uma preferência ou gosto)
            - feedback (quando o usuário fornece feedback)
            
            Exemplos:
            Entrada: "A Terra é redonda"
            Resposta: fact
            
            Entrada: "Qual é a capital do Brasil?"
            Resposta: question
            
            Entrada: "Eu prefiro explicações detalhadas"
            Resposta: preference
            
            Entrada: "Gostei muito da sua resposta"
            Resposta: feedback"""
            
            system_message = SystemMessage(content=system_content)
            human_message = HumanMessage(content=state["input"])
            
            result = self.llm.invoke([system_message, human_message])
            
            # Limpar e validar a resposta
            intent = result.content.strip().lower()
            
            # Lista de intenções válidas
            valid_intents = ["fact", "question", "preference", "feedback"]
            
            # Extrair a primeira palavra que corresponde a uma intenção válida
            intent = next((word for word in intent.split() if word in valid_intents), "question")
            
            state["intent"] = intent
            state["error"] = None
            logger.info(f"Intenção detectada: {state['intent']}")
            return state
        except Exception as e:
            error_msg = f"Erro ao processar entrada: {e}"
            logger.error(error_msg)
            state["error"] = error_msg
            return state

    def get_context(self, state: ChatState) -> ChatState:
        """Recupera contexto relevante do armazenamento vetorial"""
        try:
            logger.info("Buscando contexto relevante")
            if state["intent"] in ["question", "fact"]:
                docs = self.vector_store.similarity_search(
                    state["input"],
                    k=3
                )
                state["context"] = [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in docs
                ]
                logger.info(f"Encontrados {len(state['context'])} documentos relevantes")
            else:
                state["context"] = []
            return state
        except Exception as e:
            logger.error(f"Erro ao buscar contexto: {e}")
            state["error"] = str(e)
            return state

    def validate_fact(self, state: ChatState) -> ChatState:
        """Valida se a entrada contém um fato verificável"""
        try:
            if state["intent"] == "fact":
                logger.info("Validando fato")
                
                # Preparar contexto para validação
                context_str = ""
                if state["context"]:
                    context_str = "\n".join([
                        f"- {doc['content']}" for doc in state["context"]
                    ])
                
                system_message = SystemMessage(content="""Você é um assistente especializado em validar fatos em português.
                
                Analise cuidadosamente a entrada do usuário e determine se é uma afirmação factual que pode ser validada.
                
                Responda apenas com:
                - true: se for um fato claro e verificável
                - false: se for opinião, preferência ou não puder ser verificado
                
                Exemplos de fatos válidos:
                - "A água ferve a 100°C ao nível do mar"
                - "O Brasil é o maior país da América do Sul"
                
                Exemplos de não-fatos:
                - "Eu adoro chocolate"
                - "O azul é a cor mais bonita"
                
                Considere apenas a verificabilidade, não a veracidade.
                
                Contexto conhecido:
                {context}""")
                
                human_message = HumanMessage(content=state["input"])
                
                result = self.llm.invoke([system_message, human_message])
                
                # Extrair apenas true/false da resposta
                response = result.content.strip().lower()
                is_valid = "true" in response.split()
                
                state["is_valid"] = is_valid
                logger.info(f"Fato validado: {state['is_valid']}")
            else:
                state["is_valid"] = False
            return state
        except Exception as e:
            logger.error(f"Erro ao validar fato: {e}")
            state["error"] = str(e)
            return state

    def update_preferences(self, state: ChatState) -> ChatState:
        """Atualiza preferências do usuário com base na entrada"""
        try:
            if state["intent"] == "preference":
                logger.info("Processando atualização de preferências")
                
                # Definir valores válidos
                valid_values = {
                    "tom": ["formal", "casual"],
                    "verbosidade": ["concisa", "balanceada", "detalhada"],
                    "formalidade": ["formal", "informal"]
                }
                
                system_content = """Você é um analisador de preferências.
                
                IMPORTANTE: Sua resposta deve ser EXATAMENTE um objeto JSON válido, sem texto adicional.
                
                Analise a mensagem e identifique preferências relacionadas a:
                - tom: formal ou casual
                - verbosidade: concisa, balanceada ou detalhada
                - formalidade: formal ou informal
                
                Se nenhuma preferência for identificada, retorne {}.
                Se identificar uma preferência, inclua APENAS as preferências mencionadas.
                
                Exemplos:
                
                Entrada: "Prefiro um tom mais formal"
                Resposta: {"tom": "formal"}
                
                Entrada: "Gosto de explicações detalhadas"
                Resposta: {"verbosidade": "detalhada"}
                
                Entrada: "Quero respostas formais e concisas"
                Resposta: {"tom": "formal", "verbosidade": "concisa"}
                
                Entrada: "Gosto de matemática"
                Resposta: {}"""
                
                system_message = SystemMessage(content=system_content)
                human_message = HumanMessage(content=state["input"])
                
                # Registrar a entrada para debug
                logger.debug(f"Entrada do usuário: {state['input']}")
                
                # Obter resposta do LLM
                result = self.llm.invoke([system_message, human_message])
                
                # Registrar a resposta bruta
                logger.debug(f"Resposta bruta do LLM: {result.content}")
                
                try:
                    # Extrair apenas o JSON da resposta
                    json_str = result.content.strip()
                    if json_str.startswith('```json'):
                        json_str = json_str.replace('```json', '').replace('```', '').strip()
                    elif json_str.startswith('{'):
                        # Já está no formato JSON
                        pass
                    else:
                        # Se não encontrar JSON válido, usar objeto vazio
                        json_str = '{}'
                    
                    # Tentar fazer o parse do JSON
                    new_prefs = json.loads(json_str)
                    logger.debug(f"JSON parseado: {new_prefs}")
                    
                    # Filtrar e validar preferências
                    filtered_prefs = {}
                    for key, value in new_prefs.items():
                        if key in valid_values and isinstance(value, str):
                            value_lower = value.lower()
                            if value_lower in valid_values[key]:
                                filtered_prefs[key] = value_lower
                    
                    # Atualizar preferências mantendo valores padrão para campos não especificados
                    state["preferences"] = {**self.default_preferences, **filtered_prefs}
                    logger.info(f"Preferências atualizadas: {state['preferences']}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Erro ao decodificar JSON: {e}")
                    logger.error(f"Conteúdo que causou erro: {json_str}")
                    state["preferences"] = self.default_preferences
                except Exception as e:
                    logger.error(f"Erro ao processar preferências: {e}")
                    logger.error(f"Conteúdo que causou erro: {result.content}")
                    state["preferences"] = self.default_preferences
            else:
                state["preferences"] = self.default_preferences
            
            return state
            
        except Exception as e:
            error_msg = f"Erro ao atualizar preferências: {e}"
            logger.error(error_msg)
            state["error"] = error_msg
            return state

    def store_information(self, state: ChatState) -> ChatState:
        """Armazena informações validadas no armazenamento vetorial"""
        try:
            if state.get("error"):
                logger.warning("Pulando armazenamento de informações devido a erro anterior")
                return state

            if state["is_valid"] and state["intent"] in ["fact", "preference"]:
                logger.info("Armazenando informações validadas")
                
                # Preparar metadados
                metadata = {
                    "type": state["intent"]
                }
                
                # Se for uma preferência, converter o dicionário em string JSON
                if state["intent"] == "preference":
                    metadata["preferences"] = json.dumps(state["preferences"], ensure_ascii=False)
                
                doc = Document(
                    page_content=state["input"],
                    metadata=metadata
                )
                self.vector_store.add_documents([doc])
                logger.info("Informações armazenadas com sucesso")
            else:
                logger.info("Informações não válidas ou não armazenáveis, pulando armazenamento")
            return state
        except Exception as e:
            error_msg = f"Erro ao armazenar informações: {e}"
            logger.error(error_msg)
            state["error"] = error_msg
            return state

    def generate_response(self, state: ChatState) -> ChatState:
        """Gera uma resposta baseada no estado da conversa"""
        try:
            # Preparar contexto para a resposta
            context_str = ""
            if state["context"]:
                context_str = "\n".join([
                    f"- {doc['content']}" for doc in state["context"]
                ])

            logger.info("Gerando resposta")
            
            # Formatar preferências para o prompt
            prefs_str = "\n".join([
                f"- {key}: {value}"
                for key, value in state["preferences"].items()
            ])
            
            system_content = f"""Você é um assistente amigável em português que aprende com conversas.
            
            Analise a entrada do usuário e o contexto fornecido. Se houver informações relevantes no contexto,
            use-as para enriquecer sua resposta.
            
            Preferências do usuário:
            {prefs_str}
            
            Adapte seu tom e estilo de acordo com as preferências do usuário.
            
            Diretrizes para resposta:
            - Se for um fato: confirme se foi validado e armazenado
            - Se for uma pergunta: use o contexto para fornecer uma resposta precisa
            - Se for uma preferência: confirme as mudanças
            - Se for um feedback: agradeça e explique como isso ajuda a melhorar
            
            Mantenha suas respostas em português, de forma concisa e relevante.
            Use uma linguagem natural e amigável."""
            
            system_message = SystemMessage(content=system_content)
            
            # Preparar mensagem de contexto
            context_content = "Contexto disponível: " + context_str if context_str else "Nenhum contexto relevante disponível."
            context_message = HumanMessage(content=context_content)
            
            # Mensagem do usuário
            user_message = HumanMessage(content=state["input"])
            
            # Invocar o LLM com todas as mensagens
            result = self.llm.invoke([
                system_message,
                context_message,
                user_message
            ])
            
            state["response"] = result.content
            return state
            
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {e}")
            state["error"] = str(e)
            state["response"] = "Desculpe, ocorreu um erro ao processar sua mensagem. Por favor, tente novamente."
            return state

    def process_message(self, message: str) -> Dict:
        """Processa uma mensagem e retorna a resposta"""
        try:
            logger.info("Iniciando processamento de mensagem")
            initial_state = ChatState(
                input=message,
                intent="",
                is_valid=False,
                response="",
                error=None,
                preferences=self.default_preferences.copy(),
                context=[]
            )
            
            final_state = self.workflow.invoke(initial_state)
            
            if final_state.get("error"):
                logger.error(f"Processamento de mensagem falhou: {final_state['error']}")
                return {
                    "response": f"Desculpe, ocorreu um erro ao processar sua mensagem: {final_state['error']}",
                    "is_valid": False,
                    "error": final_state["error"],
                    "intent": final_state.get("intent", ""),
                    "preferences": final_state.get("preferences", {})
                }
            
            logger.info("Processamento de mensagem concluído com sucesso")
            return {
                "response": final_state["response"],
                "is_valid": final_state["is_valid"],
                "error": None,
                "intent": final_state["intent"],
                "preferences": final_state["preferences"]
            }
        except Exception as e:
            error_msg = f"Erro ao processar mensagem: {e}"
            logger.error(error_msg)
            return {
                "response": f"Desculpe, ocorreu um erro ao processar sua mensagem: {error_msg}",
                "is_valid": False,
                "error": str(e),
                "intent": "",
                "preferences": self.default_preferences.copy()
            }
