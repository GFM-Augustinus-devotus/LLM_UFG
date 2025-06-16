
import os
from dotenv import load_dotenv
from trulens_eval import Tru
from trulens_eval import OpenAI, Feedback, TruLlama
import numpy as np
import pandas as pd
from llm import load_qa
import random
from trulens_eval import TruChain


# Carrega variáveis do .env
load_dotenv()

# Recupera a chave da variável de ambiente
openai_key = os.getenv("OPENAI-API-KEY")

# Inicializa o provedor OpenAI com a chave
openai_provider = OpenAI(
    model_engine="gpt-3.5-turbo-1106",
    api_key=openai_key
)

df, qa = load_qa()

pasta_textos = "Textos_Extraidos"
txt_files = [
    os.path.join(pasta_textos, f)
    for f in os.listdir(pasta_textos)
    if f.lower().endswith(".txt") and os.path.getsize(os.path.join(pasta_textos, f)) > 0
]

if txt_files:
    # Escolhe um arquivo aleatório
    txt_file = random.choice(txt_files)
    with open(txt_file, "r", encoding="utf-8") as file:
        texto = file.read()

    # Pega o primeiro parágrafo não vazio
    paragrafos = [p.strip() for p in texto.split('\n') if p.strip()]
    if paragrafos:
        trecho = paragrafos[0][:1000]  # Limita o trecho a 1000 caracteres
        pergunta = f"O que diz o seguinte trecho do documento '{os.path.basename(txt_file)}'?\n\n{trecho}\n"
        print("Pergunta gerada:")
        print(pergunta)
else:
    print("Nenhum arquivo de texto encontrado em Textos_Extraidos.")

if 'Question' in df.columns and 'Answer' in df.columns:
    qa_set = [{"query": row["Question"], "response": row["Answer"]} for _, row in df.iterrows()]
else:
    # Se não houver essas colunas, adapte conforme necessário
    qa_set = []
# Initialize metrics to collect 

# Answer relevance
f_qa_relevance = Feedback(
    openai_provider.relevance_with_cot_reasons, name="Answer Relevance"
).on_input_output()

# Context relevance
f_qs_relevance = Feedback(
    openai_provider.relevance_with_cot_reasons, name="Context Relevance"
).on_input().on(TruLlama.select_source_nodes().node.text).aggregate(np.mean)


# Groundedness
f_groundedness = Feedback(
    openai_provider.groundedness_measure_with_cot_reasons, name="Groundedness"
).on(TruLlama.select_source_nodes().node.text).on_output()

# Ground truth agreement 
def groundtruth_agreement(inputs, outputs):
    query = inputs  # inputs é a string da pergunta
    resposta_modelo = outputs
    for item in qa_set:
        if item["query"] == query:
            resposta_correta = item["response"]
            return int(resposta_modelo.strip().lower() == resposta_correta.strip().lower())
    return 0
f_groundtruth = Feedback(
    groundtruth_agreement, name="Answer Correctness"
).on_input_output()


metrics = [f_qa_relevance, f_qs_relevance, f_groundedness, f_groundtruth]

def get_trulens_recorder(query_engine, app_id):
    tru_recorder = TruChain(
        app=query_engine,
        feedbacks=metrics,
        app_id=app_id,
        app_name=app_id 
    )
    return tru_recorder

tru_recorder = get_trulens_recorder(qa, app_id="QA LLM v1")

# Faça uma consulta e registre a avaliação
with tru_recorder as recording:
    try:
        resposta = qa.run(pergunta)
        print("Resposta do modelo:")
        print(resposta)
    except Exception as e:
        print("Erro ao consultar o modelo:", e)

# Obtenha e mostre o leaderboard com as métricas
tru = Tru()
leaderboard = tru.get_leaderboard()
print(leaderboard.columns)
print(leaderboard)
#print(leaderboard[["app_id", "record_id", "Groundedness", "Answer Relevance", "Context Relevance", "Answer Correctness"]])

tru.run_dashboard()