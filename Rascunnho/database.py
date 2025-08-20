from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import tiktoken

load_dotenv()

DIRETORIO = "Documentos_Importantes"

def tiktoken_len(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def criar_db():
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)

def carregar_documentos():
    loader = PyPDFDirectoryLoader(DIRETORIO, glob="*.pdf")
    return loader.load()

def dividir_chunks(documentos):
    # dividir por TOKENS, não por caracteres
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,          # ~800 tokens por chunk (bom para embeddings)
        chunk_overlap=200,       # sobreposição segura
        length_function=tiktoken_len,
        add_start_index=True
    )
    chunks = splitter.split_documents(documentos)
    print(f"Total de chunks: {len(chunks)}")
    return chunks

def vetorizar_chunks(chunks):
    # Atenção: aqui "chunk_size" = tamanho do LOTE por requisição à API de embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        chunk_size=50            
    )
    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="DataBase",
        collection_name="meus_docs"
    )

    print("Banco de dados criado!")

criar_db()
