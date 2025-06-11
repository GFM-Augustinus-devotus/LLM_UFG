#-----------------------------
from llm import *
from llm import df
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from trulens.apps.langchain import TruChain

from trulens_eval.feedback.provider.openai import OpenAI as TruOpenAI

from trulens_eval import Feedback, Select, TruCustomApp
from trulens_eval.feedback.provider.openai import OpenAI
from trulens_eval.tru_custom_app import instrument

from trulens_eval import Tru, Feedback, Select, TruCustomApp
from trulens_eval.tru_custom_app import instrument
from trulens_eval.feedback.provider.openai import OpenAI
from trulens_eval.guardrails.base import context_filter
from trulens_eval.utils.display import get_feedback_result
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

for idx, row in df.iterrows():
    vetor.add(
        ids=[str(idx)],  # ID único para cada documento
        documents=[row['text']],
        metadatas=[{"page_name": row['page_name']}]
    )


#--------------------------------------------

# Criando o RAG
# O instrument irá entrar como um medidor da llm.
# No caso serão 3 parâmetros Groundness | Answer Relevance | Context Relevance

tru = Tru()
tru.reset_database() #irá reiniciar a base de dados será passada para o Trulens

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

#Feedback functions servem para avaliar e registrar o que a LLM fez, 
# como ela se comportou, e como sua resposta se relaciona com a entrada — com o objetivo de:
#  Melhorar resultados Detectar erros Ajustar estratégias (prompting, ranking, etc.) Logar interações para análise posterior

provedor = OpenAI()

#Analisando o Groundness

groundness = (
    Feedback(provedor.groundness_measure_with_cot_reasons, name = "Groundness").on(Select.RecordCalls.retrieve.rets.collect()).on_output()
)

#Analsando a Relevancia das respostas

respostas = (Feedback(provedor.relevance_with_cot_reasons, name = "Relevância de Respostas").on_input().on_output())

#Analisando a relevância do contexto

contexto = (Feedback(provedor.context_relevance_with_cot_reasons, name = "Relevância de Contexto").on_input().on(Select.RecordCalls.retrieve.rets[:]).aggregate(np.mean))

#Customizando o RAG com o TruCustomApp, e adicionando a lista de Feedbacks para a avaliação

tru_rag = TruCustomApp(rag, app_id = "RAG v1", feedbacks = [groundness,  respostas, contexto])


#Para rodar o aplicativo

with tru_rag as recording:
    rag.query("O que o Plano de Desenvolvimento Institucional da UFG está explicando?")

tru.get_leaderboard() #Dashboard Do Trulens para ver o desempenho do arquivo


#----------------------- Criando uma segunda versão do App com os GuardRails
#Usando GuardRails para melhorar a forma de análise desempenho do APP

# contexto_pontuacao = (Feedback(provedor.contexto, name = "Relevância de contexto").on_input().on(Select.RecordCalls.Retrieve.rets))

# class RAG_Com_Filtros:
#     @instrument
#     @context_filter(contexto_pontuacao, 0.5) #Vai filtrar a parti de pontuações de 0.5
#     def abstrair(self, query: str) -> list:  #Retirnado a relevância do texto
#         resultado = vetor.query(query_texts=query,n_results=4)
#         return [documento for sublista in resultado['documents'] for documento in sublista]
    
#     @instrument
#     def responder(self, query: str, contexto_str:list) -> str: #Gerar uma resposta a partir do contexto

#         resultado = openai_client.chat.completions.create(
#                 model = "gpt-3.5-turbo",
#                 temperature = 0, 
#                 messages = [{"role": "user",
#                             "content": 
#                             f"Foi passado o contexto informativo logo abaixo. \n"
#                             f"\n--------------------------\n"
#                             f"{contexto_str}"
#                             f"\n--------------------------\n"
#                             f"Dado essa informação, por favor responda a pergunta:{query}"
#                             }]
#             ).choices[0].message.content
#         return resultado

#     def query(self, query: str)-> str:
#         abstracao = self.retrieve(query)
#         completition = self.generate_completition(query, abstracao)
#         return abstracao
    
# rag = RAG_Com_Filtros()

# Tru_Rag_Filtrado = TruCustomApp(rag , app_id = 'RAG v2', feedbacks = [groundness, respostas, contexto])

# with Tru_Rag_Filtrado as recording:
#     rag.query("Fale sobre a marca escolhida pela UFG como sua marca oficial, e por qual motivo ela escolheu esse tipo de marca?")

# tru.get_leaderboard(app_ids=[])

# resultados_finais = recording.records[-1]
# get_feedback_result(resultados_finais, "Relevância de Contexto")









