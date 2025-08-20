from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from ragas.metrics import AnswerCorrectness, ContextRelevance, Faithfulness
from ragas import evaluate
from datasets import Dataset
import os, re, hashlib


load_dotenv()

PERSIST_DIR = "DataBase"
COLLECTION = "meus_docs"
EMBED_MODEL = "text-embedding-3-small"

SYSTEM_MSG = (
    "Você é um assistente de RAG. Regras:\n"
    "1) Responda APENAS com base no contexto fornecido.\n"
    '2) Se a resposta não estiver no contexto, diga exatamente: "Não sei a resposta.".\n'
    "3) Responda uma ÚNICA vez, de forma concisa, sem repetir frases ou parágrafos.\n"
    "4) Estruture em tópicos curtos quando houver etapas/regras.\n"
    "5) Inclua fontes citando arquivo e página quando possível."
)
HUMAN_TEMPLATE = (
    "Pergunta:\n{pergunta}\n\n"
    "Contexto:\n{base_conhecimento}\n"
)

def normalize_para(p: str) -> str:
    p = re.sub(r"\s+", " ", p.strip().lower())
    return p

def dedupe_paragraphs(textos, max_chars=8000):
    seen = set()
    out = []
    total = 0
    for bloco in textos:
        for p in re.split(r"\n{2,}", bloco.strip()):
            np = normalize_para(p)
            h = hashlib.md5(np.encode()).hexdigest()
            if np and h not in seen:
                cand = p.strip()
                if total + len(cand) > max_chars:
                    return "\n\n----\n\n".join(out)
                out.append(cand)
                total += len(cand)
                seen.add(h)
    return "\n\n----\n\n".join(out)

def dedupe_lines(text: str) -> str:
    seen = set()
    out = []
    for line in text.splitlines():
        key = line.strip()
        if key and key not in seen:
            out.append(line)
            seen.add(key)
        elif not key:
            if out and out[-1] != "":
                out.append("")
    return "\n".join(out).strip()

def normalize_score_to_01(score):
    if 0.0 <= score <= 1.0:
        return score
    return (score + 1.0) / 2.0

def perguntar():
    pergunta = input("Escreva sua pergunta:\n")

    funcao_embedding = OpenAIEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=funcao_embedding,
        collection_name=COLLECTION
    )

    resultados = db.similarity_search_with_relevance_scores(pergunta, k=6)

    if not resultados:
        print("Nenhum resultado retornado pelo índice vetorial.")
        print('Resposta da IA:\nNão sei a resposta.')
        return

    pares_doc_score = []
    for doc, score in resultados:
        score_norm = normalize_score_to_01(score)
        pares_doc_score.append((doc, score_norm))

    THRESHOLD = 0.40
    filtrados = [(d, s) for d, s in pares_doc_score if s >= THRESHOLD]

    if not filtrados:
        print(f"Nenhum trecho atingiu o nível mínimo de relevância (threshold={THRESHOLD}).")
        print('Resposta da IA:\nNão sei a resposta.')
        return

    # monta blocos com fonte/página
    textos = []
    for doc, sc in sorted(filtrados, key=lambda x: x[1], reverse=True):
        fonte = doc.metadata.get("source", "desconhecida")
        pagina = doc.metadata.get("page", None)
        header = f"[Fonte: {fonte}" + (f", página: {pagina}]" if pagina is not None else "]")
        textos.append(header + "\n" + doc.page_content.strip())

    base_conhecimento = dedupe_paragraphs(textos, max_chars=8000)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MSG),
        ("human", HUMAN_TEMPLATE),
    ])
    final_prompt = prompt.invoke({
        "pergunta": pergunta,
        "base_conhecimento": base_conhecimento
    })

    modelo = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0.1,
        max_tokens=400,
        frequency_penalty=0.2,
    )
    resposta = modelo.invoke(final_prompt).content
    resposta = dedupe_lines(resposta)

    print("DEBUG - Top resultados (normalizados):")
    for i, (doc, s) in enumerate(filtrados, start=1):
        print(f"  {i:02d}) score={s:.3f} | source={doc.metadata.get('source')} | page={doc.metadata.get('page')}")

    print("\nResposta da IA:\n", resposta)

#Criando Métricas RAGAS para o rag do Bot UFG
    ragas_contexts = [doc.page_content for doc, score in filtrados if doc.page_content and doc.page_content.strip()]

    resposta = modelo.invoke(final_prompt).content
    resposta = dedupe_lines(resposta)

    ground_truth = os.getenv("EVAL_GROUND_TRUTH", "").strip()

    metrics = [ContextRelevance(), Faithfulness()]
    if ground_truth:
        metrics.append(AnswerCorrectness())

    data_dict = {
        "question": [pergunta],
        "answer": [resposta],
        "contexts": [ragas_contexts]
    }

    if ground_truth:
        try:
            metrics = [AnswerCorrectness, ContextRelevance, Faithfulness]
            data_dict["ground_truth"] = [ground_truth]
        except NameError:
            print("Erro ao calcular o valor de ground_truth")
            pass
    eval_ds = Dataset.from_dict(data_dict)

    eval_result = evaluate(eval_ds, metrics=metrics)

    print("\nMétricas do RAGAS:")

    try:
        df = eval_result.to_pandas()  
    except AttributeError:
        try:
            df = eval_result.to_dataframe()
        except AttributeError:
            df = None

    try:
        df = eval_result.to_pandas()
    except AttributeError:
        df = eval_result.to_dataframe()

    
    rename_map = {c: c.replace("nv_", "") for c in df.columns if c.startswith("nv_")}
    df = df.rename(columns=rename_map)

    row = df.iloc[0]

    if df is not None:
        
        row = df.iloc[0]
        for metric_name, value in row.items():
            try:
                print(f"{metric_name}: {float(value):.3f}")
            except Exception:
                print(f"{metric_name}: {value}")
    else:
        
        try:
            d = eval_result.to_dict() 
            for k, v in d.items():
                try:
                    val = v[0] if isinstance(v, list) and v else v
                    print(f"{k}: {float(val):.3f}")
                except Exception:
                    print(f"{k}: {val}")
        except Exception:
            print(str(eval_result))

perguntar()



