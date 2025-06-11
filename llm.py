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
import pdfplumber
import re

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

#Extraindo mais textos (No caso são os PDFs da pasta de documentos importantes)
diretório = "Documentos_Importantes"
textos = {}

def clean_pdf_text(text):
    text = re.sub(r'\n{2,}', '[PARAGRAPH]', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\[PARAGRAPH\]' , '\n\n', text)
    text = re.sub(r' +' , ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

for filename in os.listdir(diretório):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(diretório, filename)
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            text = clean_pdf_text(text)
            textos[filename] = text
            txt_filename = filename.replace(".pdf",  ".txt")
            with open(txt_filename, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Texto extraído e salvo: {txt_filename}")
        except Exception as e:
            print(f"Erro ao extrair o texto: {filename}: {e}")

# for filename in os.listdir(diretório):
#     if filename.lower().endswith(".pdf"):
#         pdf_path = os.path.join(diretório, filename)
#         text = ""
#         try:
#             reader = PdfReader(pdf_path)
#             for page in reader.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
#             textos[filename] = text
#             # Salva cada PDF como um .txt para uso posterior
#             txt_filename = filename.replace(".pdf", ".txt")
#             with open(txt_filename, "w", encoding="utf-8") as f:
#                 f.write(text)
#             print(f"Extraído e salvo: {txt_filename}")
#         except Exception as e:
#             print(f"Erro ao processar {filename}: {e}")

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

txt_files = ["Estatuto.txt", "RGCG.txt", "Regimento.txt"]

# Adiciona os arquivos .txt gerados a partir dos PDFs da pasta Documentos_Importantes
for filename in os.listdir(diretório):
    if filename.lower().endswith(".pdf"):
        txt_filename = filename.replace(".pdf", ".txt")
        if os.path.exists(txt_filename) and txt_filename not in txt_files:
            txt_files.append(txt_filename)

# Monta o DataFrame com todos os textos
data = {
    'page_name': [],
    'text': []
}

for txt_file in txt_files:
    with open(txt_file, "r", encoding="utf-8") as file:
        text = file.read()
        data['page_name'].append(txt_file)
        data['text'].append(text)    

df = pd.DataFrame(data)
print(df)

#Gerando um arquivo tabular para o data frame
# df.to_csv('/content/scraped.csv', escapechar='\\')
# df.head()

#Carregando o dataset basedao nos textos extraídos


dataset = load_dataset(
    'text',
    data_files={'train': txt_files}
)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# documents = [Document(page_content=text) for text in dataset['train']['text']]

print(dataset['train'].column_names)  # Adicione para depuração

# Ajuste para garantir que está pegando o campo correto
if 'text' in dataset['train'].column_names:
    documents = [Document(page_content=text) for text in dataset['train']['text'] if len(text.strip()) > 100]
else:
    # Tente pegar o primeiro campo disponível
    first_col = dataset['train'].column_names[0]
    documents = [Document(page_content=text) for text in dataset['train'][first_col] if len(text.strip()) > 100]

texts = text_splitter.split_documents(documents)
print(f"Total de Cunks: {len(texts)}")

embeddings = OpenAIEmbeddings(openai_api_key=key)
db = FAISS.from_documents(texts, embeddings)

#Construindo o Prompt padrão que será utilizado

from langchain.chains import RetrievalQA
from langchain import PromptTemplate

retriever = db.as_retriever(search_kwargs={"k": 1})
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

# print("Arquivos carregados:", txt_files)
# docs = retriever.get_relevant_documents("Teste de pergunta")
# print(f"Chunks retornados: {len(docs)}")
# print(f"Tamanho total dos textos: {sum(len(doc.page_content) for doc in docs)} caracteres")

# for i, chunk in enumerate(texts):
#     print(f"Chunk {i} ({len(chunk.page_content)} chars): {repr(chunk.page_content[:100])}")