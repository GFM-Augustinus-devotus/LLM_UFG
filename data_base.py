# database.py
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import tiktoken
import os

load_dotenv()

DIRETORIO = "Documentos_Importantes"
PERSIST_DIR = "DataBase"
COLLECTION = "meus_docs"
EMBED_MODEL = "text-embedding-3-small" 

def tiktoken_len(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def carregar_documentos():
    loader = PyPDFDirectoryLoader(DIRETORIO, glob="*.pdf")
    return loader.load()

def dividir_chunks(documentos):
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       
        chunk_overlap=120,    
        length_function=tiktoken_len,
        add_start_index=True
    )
    chunks = splitter.split_documents(documentos)
    print(f"Total de chunks: {len(chunks)}")
    return chunks

def criar_db():
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)

    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        chunk_size=50  
    )

    Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION
    )
    print(f"Banco de dados criado/atualizado em '{PERSIST_DIR}', coleção '{COLLECTION}'.")

if __name__ == "__main__":
    criar_db()
