import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from trulens_eval import Tru, Feedback, Select, TruLlama
from trulens_eval.feedback.provider.openai import OpenAI as TruOpenAI

# 1. Carregar chave da OpenAI
load_dotenv()
key = os.getenv("OPENAI-API-KEY")

# 2. Inicializar LLM para LlamaIndex
llm = LlamaOpenAI(api_key=key, model="gpt-3.5-turbo")

# 3. Carregar documentos (supondo que você já tem os arquivos .txt)
documents = SimpleDirectoryReader(input_dir="./", required_exts=[".txt"]).load_data()

# 4. Inicializar ChromaDB e o VectorStore
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("ufg_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 5. Criar o índice
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

# 6. Criar o engine de consulta
query_engine = index.as_query_engine(llm=llm)

# 7. Integrar com Trulens para avaliação
tru = Tru()
tru_openai = TruOpenAI()

# Exemplos de feedbacks
groundness = Feedback(tru_openai.groundedness_measure_with_cot_reasons, name="Groundedness").on(Select.Context).on_output()
relevance = Feedback(tru_openai.relevance_with_cot_reasons, name="Relevance").on_input_output()

# Envolvendo o query_engine com o TruLlama para avaliação
tru_query_engine = TruLlama(
    query_engine,
    app_id="LlamaIndex-RAG-UFG",
    feedbacks=[groundness, relevance]
)

# 8. Fazer uma pergunta e avaliar
with tru_query_engine as recording:
    pergunta = "O que o Plano de Desenvolvimento Institucional da UFG apresenta?"
    resposta = query_engine.query(pergunta)
    print("Resposta:", resposta)

# 9. Visualizar resultados no dashboard do Trulens
tru.run_dashboard()  # Isso abrirá o dashboard em http://localhost:8501
