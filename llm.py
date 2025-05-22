 #Construindo o DataSet da UFG
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datasets import load_dataset
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
import os

#Carregando a chave da API da OpenAI

load_dotenv()
key = os.getenv("OPENAI-API-KEY")

#Estatuto UFG e documentos relacionados
link = "https://ufg.br/p/6383-documentos"
headers = {'User-Agent': "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36" }

#Extraindo o Header do estatutoz
page = requests.get(link, headers=headers)
soup = BeautifulSoup(page.text, 'html.parser')

#Extrair os títulos do Estatuto

pesquisa = soup.find_all("h1")
EstatutoTitulo = ""
for texto in pesquisa:
    EstatutoTitulo += texto.get_text() + "\n"
#print(EstatutoTitulo)

#Extrair os textos do Estatuto
EstatutoTexto = ""
pesquisa = soup.find_all("p")
#print(pesquisa)

for texto in pesquisa:
    EstatutoTexto += texto.get_text() + "\n"

#print(EstatutoTexto)

#Regulamento Geral dos Cursos de Graduação

url = "https://files.cercomp.ufg.br/weby/up/765/o/rgcg.pdf"
RGCG_Titulo = "RGCG.pdf"
RGCGtextos = ""

response = requests.get(url)
with open(RGCG_Titulo, "wb") as f:
    f.write(response.content)

reader = PdfReader(RGCG_Titulo)
i = 0
for i in range(len(reader.pages)):
    page = reader.pages[i]
    RGCGtextos = page.extract_text()
    #print(RGCGtextos)

#print("------------------------------------------")

#Retirar partes desnecessárias de forma manual do PDF
PontoInicio = "I-"
PontoFinal = "X-"
IndiceInicio = RGCGtextos.find(PontoInicio)
IndiceFinal = RGCGtextos.find(PontoFinal)

#Verificação se os pontos selecionados existem no texto extraído
if IndiceInicio != -1 and IndiceFinal != -1:
    print("Ponto de início" + str(IndiceInicio))
    print("Ponto final" + str(IndiceFinal))
    RGCGtextos = RGCGtextos[1 : IndiceInicio] + RGCGtextos[IndiceFinal : len(RGCGtextos)]
    #print(RGCGtextos)
else:
    print("Pontos de início e fim não encontrados no texto.")

#Regimento Interno

url = "https://files.cercomp.ufg.br/weby/up/66/o/Regimento_UFG.pdf"
RegimentoTitulo = "Regimento.pdf"
RegimentoTextos = ""

response = requests.get(url)
with open(RegimentoTitulo, "wb") as f:
    f.write(response.content)


reader = PdfReader(RegimentoTitulo)
i = 0
for i in range(len(reader.pages)):
    page = reader.pages[i]
    RegimentoTextos = page.extract_text()
    #print(RegimentoTextos)

#Gerando arquivo do estatuto
with open("Estatuto.txt", "w", encoding="utf-8") as file:
    file.write(EstatutoTitulo + "\n")
    file.write(EstatutoTexto + "\n")
if file.closed:
    print("Arquivo criado com sucesso!")

#Gerando arquivo do RGCG
with open("RGCG.txt", "w", encoding="utf-8") as file:
    file.write(RGCGtextos + "\n")
if file.closed:
    print("Arquivo criado com sucesso!")

#Gerando arquivo do Regimento
with open("Regimento.txt", "w", encoding="utf-8") as file:
    file.write(RegimentoTextos + "\n")
if file.closed:
    print("Arquivo criado com sucesso!")

#Gerando o DataFrame
data = {
    'page_name': ['Estatuto.txt' ,'RGCG.txt', 'Regimento.txt'],
    'text': []
}

with open("Estatuto.txt", "r", encoding="utf-8") as file:
    text1 = file.read()
    data['text'].append(text1)

with open("RGCG.txt", "r", encoding="utf-8") as file:
    text2 = file.read()
    data['text'].append(text2)

with open("Regimento.txt", "r", encoding="utf-8") as file:
    text3 = file.read()
    data['text'].append(text3)

df = pd.DataFrame(data)
print(df)

#Gerando um arquivo tabular para o data frame
# df.to_csv('/content/scraped.csv', escapechar='\\')
# df.head()

#Carregando o dataset basedao nos textos extraídos


dataset = load_dataset(
    'text',
    data_files={'train': ['Estatuto.txt', 'Regimento.txt', 'RGCG.txt']}
)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# documents = [Document(page_content=text) for text in dataset['train']['text']]

print(dataset['train'].column_names)  # Adicione para depuração

# Ajuste para garantir que está pegando o campo correto
if 'text' in dataset['train'].column_names:
    documents = [Document(page_content=text) for text in dataset['train']['text']]
else:
    # Tente pegar o primeiro campo disponível
    first_col = dataset['train'].column_names[0]
    documents = [Document(page_content=text) for text in dataset['train'][first_col]]

texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=key)
db = FAISS.from_documents(texts, embeddings)

#Construindo o Prompt padrão que será utilizado

from langchain.chains import RetrievalQA
from langchain import PromptTemplate

retriever = db.as_retriever()
prompt_template = """Use as seguintes informações contextuais para responder à pergunta no final. Se você não souber a resposta, diga: "Não sei".

Contexto: {context}

Pergunta: {question}"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}
llm = ChatOpenAI(openai_api_key=key, model="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)

#Gerando as perguntas relacionadas ao dataset selecionado

# query1 = "O que o Plano de Desenvolvimento Institucional da UFG apresenta?"
# resultado1 = qa.invoke({"query": query1})


# query2 = "Fale sobre a marca escolhida pela UFG como sua marca oficial, e por qual motivo ela escolheu esse tipo de marca?"
# resultado2 = qa.invoke({"query": query2})

# query3 = "O que a Portaria nº 2.569, de 26 de agosto de 2020 explica expecificamente, o que eu me devo atentar sobre ela?"
# resultado3 = qa.invoke({"query": query3})

# query4 = "Em que ano entrou em vigor o novo documento do Regimento da UFG e para que ele serve?"
# resultado4 = qa.invoke({"query": query4})

# #Mostrando as respostas geradas pelo modelo
# print("------------------------")
# print("------------------------")
# print(resultado1)
# print("------------------------")
# print("------------------------")

# print("------------------------")
# print("------------------------")
# print(resultado2)
# print("------------------------")
# print("------------------------")

# print("------------------------")
# print("------------------------")
# print(resultado3)
# print("------------------------")
# print("------------------------")

# print("------------------------")
# print("------------------------")
# print(resultado4)
# print("------------------------")
# print("------------------------")