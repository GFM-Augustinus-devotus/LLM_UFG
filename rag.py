#-----------------------------
from llm import *
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from trulens_eval import Tru
from trulens_eval.tru_custom_app import instrument 
#----------------------


load_dotenv()
key = os.getenv("OPENAI-API-KEY")

#--------------------------------

#Criando um armazenamento para vetores
embedding_function = OpenAIEmbeddingFunction(api_key=key, model_name="text-embedding-ada-002")

cliente = chromadb.Client()
vetor = cliente.get_or_create_collection(name = "UFG", embedding_function="embedding_function")

#Inserindo os vetor com os valores dos dados textuais

vetor.add("Etatuto", EstatutoTexto)
vetor.add("Regimento", RegimentoTextos)
vetor.add("RGCG", RGCGtextos)

#--------------------------------------------

# Criando o RAG
# O instrument irá entrar como um medidor da llm.
# No caso serão 3 parâmetros Groundness | Answer Relevance | Context Relevance

tru = Tru()
tru.reset_database() #irá reiniciar a base de dados

openai_client = OpenAI()

class Rag_llm_ufg:
    @instrument
    #Função para abstrair informções importantes do texto
    def abstrair(self, query:str) -> list:
        
        resultado = vetor.query(
            query_texts=query,
            n_results=4         #Os 4 documentos vetoriais mais importantes
        )
    
        #Armazenar em uma única lista
        return [documento for sublista in resultado['documents'] for documento in sublista]
    
    @instrument
    #Gerar respostas a partir do contexto
    def responder(self , query:str, contexto_str:list) -> str:

        resultado = openai_client.chat.completions.create(
            model = "gpt-3.5-turbo",
            temperature = 0, 
            messages = [{"role": "user",
                         "content": 
                         f"Foi passado o contexto informativo logo abaixo. \n"
                         f"\n--------------------------\n"
                         f"{contexto_str}"
                         f"\n--------------------------\n"
                         f"Dado essa informação, por favor responda a pergunta:{query}"
                         }]
        ).choices[0].message.content
        return resultado
    
    @instrument
    #Montagem da query
    def query(self, query: str)->str:
        contexto_str = self.abstrair(query)
        abstracao = self.responder(query, contexto_str)
        return abstracao
    
rag = Rag_llm_ufg()
        









