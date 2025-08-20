import os
import pandas as pd
import tiktoken
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
import openai
import time
from dotenv import load_dotenv

# Carregar chave da OpenAI do .env
load_dotenv(dotenv_path="c:/Users/Gabriel Melo/OneDrive/Documents/all/UFG/PFC/LLM_UFG/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

tokenizer = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 8191

def extrair_textos_pdf(pasta_pdfs):
    textos = []
    for arquivo in os.listdir(pasta_pdfs):
        if arquivo.lower().endswith('.pdf'):
            caminho_pdf = os.path.join(pasta_pdfs, arquivo)
            try:
                reader = PdfReader(caminho_pdf)
                texto = ""
                for page in reader.pages:
                    texto += page.extract_text() or ""
                titulo = os.path.splitext(arquivo)[0]
                textos.append({'title': titulo, 'text': texto})
            except Exception as e:
                print(f"Erro ao processar {arquivo}: {e}")
    return textos

def dividir_em_blocos(texto, max_tokens=MAX_TOKENS):
    tokens = tokenizer.encode(texto)
    blocos = []
    for i in range(0, len(tokens), max_tokens):
        bloco_tokens = tokens[i:i+max_tokens]
        bloco_texto = tokenizer.decode(bloco_tokens)
        blocos.append(bloco_texto)
    return blocos

def gerar_embedding(texto, modelo="text-embedding-ada-002"):
    blocos = dividir_em_blocos(texto)
    embeddings = []
    for bloco in blocos:
        try:
            response = openai.embeddings.create(input=bloco, model=modelo)
            embeddings.append(response.data[0].embedding)
            time.sleep(1)  # Evita rate limit
        except Exception as e:
            print(f"Erro ao gerar embedding: {e}")
            embeddings.append([0.0] * 1536)
    # Média dos embeddings dos blocos
    if embeddings:
        embedding_medio = [float(sum(x))/len(x) for x in zip(*embeddings)]
        return embedding_medio
    else:
        return [0.0] * 1536

# --- Uso ---
pasta_pdfs = "Documentos_Importantes"

textos = extrair_textos_pdf(pasta_pdfs)
df = pd.DataFrame(textos)

docs_desejados = [
    "Resolucao_CONSUNI_2025_0324",
    "Resolucao_Monitoria_EMC_2024_v2.4_assinado"
]
df = df[df['title'].isin(docs_desejados)].reset_index(drop=True)

df['n_tokens'] = df['text'].apply(lambda x: len(tokenizer.encode(x)) if isinstance(x, str) else 0)

df.to_csv('static/tokens_por_texto.csv', index=False)

df.hist(column='n_tokens')
plt.title('Distribuição do número de tokens por texto')
plt.xlabel('Número de tokens')
plt.ylabel('Frequência')
plt.savefig('static/hist_tokens.png')
plt.close()

# Gerar embeddings
embeddings = []
for texto in df['text']:
    emb = gerar_embedding(texto)
    embeddings.append(emb)

df['embedding'] = embeddings

df.to_csv('static/embeddings_por_texto.csv', index=False)

try: 
    for texto in df['text']:
        print("Texto para embedding:", texto[:200])  # Mostra início do texto
        print("Tokens:", len(tokenizer.encode(texto)))
        emb = gerar_embedding(texto)
        print("Embedding gerado (primeiros 10):", emb[:10])
        embeddings.append(emb)

except Exception as e:
    print(f"Erro ao gerar embedding: {e}")
    embeddings.append([0.0] * 1536)

# Exemplo: mostrar 10 valores do meio do embedding
# for idx, row in df.iterrows():
#     emb = row['embedding']
#     if isinstance(emb, list) and len(emb) > 0:
#         meio = len(emb) // 2
#         print(f"{row['title']} | Tokens: {row['n_tokens']} | Embedding: {emb}")
#     else:
#         print(f"{row['title']} | Tokens: {row['n_tokens']} | Embedding não gerado")