#-----------------------------
from llm import load_qa
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from trulens.apps.langchain import TruChain

from trulens_eval.feedback import Feedback
from trulens_eval import Select, TruCustomApp, Tru, instrument

import numpy as np
from dotenv import load_dotenv
import os
#----------------------

load_dotenv()
key = os.getenv("OPENAI-API-KEY")
df , qa = load_qa()
#--------------------------------

# Criando um armazenamento para vetores
embedding_function = OpenAIEmbeddingFunction(api_key=key, model_name="text-embedding-ada-002")

cliente = chromadb.Client()
vetor = cliente.get_or_create_collection(name="UFG", embedding_function=embedding_function)

# Adicionando todos os textos do DataFrame ao vetor
for idx, row in df.iterrows():
    vetor.add(
        ids=[str(idx)],  # ID único para cada documento
        documents=[row['text']],
        metadatas=[{"page_name": row['page_name']}]
    )

#--------------------------------------------

# Criando o RAG
tru = Tru()
tru.reset_database()  # irá reiniciar a base de dados para o Trulens

openai_client = OpenAI()

class Rag_llm_ufg:
    @instrument
    def abstrair(self, query: str) -> list:
        resultado = vetor.query(
            query_texts=query,
            n_results=4  # Os 4 documentos vetoriais mais importantes
        )
        return [documento for sublista in resultado['documents'] for documento in sublista]

    @instrument
    def responder(self, query: str, contexto_str: list) -> str:
        resultado = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            temperature=0,
            messages=[{
                "role": "user",
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
    def query(self, query: str) -> str:
        contexto_str = self.abstrair(query)
        abstracao = self.responder(query, contexto_str)
        return abstracao

rag = Rag_llm_ufg()

# Feedback functions
provedor = OpenAI()

groundness = (
    Feedback(provedor.groundness_measure_with_cot_reasons, name="Groundness")
    .on(Select.RecordCalls.retrieve.rets.collect()).on_output()
)

respostas = (
    Feedback(provedor.relevance_with_cot_reasons, name="Relevância de Respostas")
    .on_input().on_output()
)

contexto = (
    Feedback(provedor.context_relevance_with_cot_reasons, name="Relevância de Contexto")
    .on_input().on(Select.RecordCalls.retrieve.rets[:]).aggregate(np.mean)
)

tru_rag = TruCustomApp(rag, app_id="RAG v1", feedbacks=[groundness, respostas, contexto])

with tru_rag as recording:
    rag.query("O que o Plano de Desenvolvimento Institucional da UFG está explicando?")

tru.get_leaderboard()