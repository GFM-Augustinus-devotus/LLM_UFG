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

def load_qa():

    load_dotenv()
    key = os.getenv("OPENAI-API-KEY")
    pasta_textos = "Textos_Extraidos"
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
            txt_filename = filename.replace(".pdf", ".txt")
            txt_path = os.path.join(pasta_textos, txt_filename)
            if os.path.exists(txt_path):
                print(f"Arquivo já existe  {txt_path}")
                continue
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
                # txt_filename = filename.replace(".pdf", ".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Texto extraído e salvo: {txt_filename}")
            except Exception as e:
                print(f"Erro ao extrair o texto: {filename}: {e}")

    #Gerando o DataFrame
    txt_files = [os.path.join(pasta_textos, "Estatuto.txt"),
                 os.path.join(pasta_textos, "RGCG.txt"),
                 os.path.join(pasta_textos, "Regimento.txt")]

    os.makedirs(pasta_textos, exist_ok=True)

    # Lista todos os arquivos .txt na pasta Textos_Extraidos
    txt_files = [
        os.path.join(pasta_textos, f)
        for f in os.listdir(pasta_textos)
        if f.lower().endswith(".txt") and os.path.getsize(os.path.join(pasta_textos, f)) > 0
    ]

    if not txt_files:
        print("Nenhum arquivo .txt válido encontrado em Textos_Extraidos.")
        return None

    # Adiciona os arquivos .txt gerados a partir dos PDFs da pasta Documentos_Importantes
    for filename in os.listdir(diretório):
        if filename.lower().endswith(".pdf"):
            txt_filename = filename.replace(".pdf", ".txt")
            txt_path = os.path.join(pasta_textos, txt_filename)
            if os.path.exists(txt_path) and txt_path not in txt_files:
                txt_files.append(txt_path)

    # Monta o DataFrame com todos os textos
    data = {
        'page_name': [],
        'text': []
    }

    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as file:
            text = file.read()
            data['page_name'].append(os.path.basename(txt_file))
            data['text'].append(text) 

    df = pd.DataFrame(data)
    print(df)

    dataset = load_dataset(
        'text',
        data_files={'train': txt_files}
    )

    text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=50)

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
    llm = ChatOpenAI(openai_api_key=key, model="gpt-4-turbo")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)

    return df , qa

if __name__ == "__main__":
    load_qa()