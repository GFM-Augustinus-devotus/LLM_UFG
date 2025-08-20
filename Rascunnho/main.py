#LLM-UFG ---> Large Language Model Aplicada na Organização de Resoluções Públicas
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
#----------------Bibliotecas--------------------------
from langchain.schema import StrOutputParser
from langchain_core.messages import SystemMessage , HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
#----------------------------------------------------

DATA_BASE = "DataBase"

load_dotenv()
key = os.getenv("OPENAI-API-KEY")

prompt_template = """
Responda a pergunta do usuário: 
{pergunta}
 com base nessas informações:
{base_conhecimeto}
Se você não encontrar a resposta para a pergunta do usuário nessas informações responda: Não sei a resposta."""

def perguntar():
    pergunta = input("Escreva sua pergunta: \n")

    #Carregar Banco de dados
    funcao_embedding = OpenAIEmbeddings()
    db = Chroma(persist_directory=DATA_BASE, embedding_function=funcao_embedding)

    #Comparar a pergunta do usuário (embedding) com o banco de dados. Diferentes tipos de similiarity search. Posso Filtrar um valor padrão que eu quero receber do nível de relevância das respostas 
    resultados = db.similarity_search_with_relevance_scores(pergunta, k=3) #k refere-se ao n° de Chunks

    if len(resultados) == 0 or resultados[0][1] < 0.7:
        print("Não escontrou nenhuma informação impotante no banco de dados")
        return
    
    textos_resultados = []
    for resultado in resultados:
        texto = resultado[0].page_content
        textos_resultados.append(texto)

    base_conhecimento = "\n\n ---- \n\n".join(textos_resultados)

    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({"pergunta": pergunta, "base_conhecimento": base_conhecimento})
    modelo = ChatOpenAI()
    resposta = modelo.invoke(prompt).content
    print("Resposta da IA: \n" , resposta)

perguntar()
    # print(prompt)
    # print(resultados[0])
    # print(len(resultados))









#-------------------------------------------
#TRASLETOR
# Mensagens = [
#     SystemMessage("Traduza o texto para o português"),
#     HumanMessage("Die Eisenfaust am Lanzenschaft")
# ]

# modelo = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key = key)
# parser = StrOutputParser()
# chain = modelo | parser

# resposta = chain.invoke(Mensagens)
# print(resposta)