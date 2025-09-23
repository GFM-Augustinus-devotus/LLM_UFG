import csv
import pandas as pd
import matplotlib.pyplot as plt

dados = []
with open("Métricas RAG OpenAI - Página1.csv", newline="", encoding="utf-8") as csvfile:
    leitor = csv.reader(csvfile, delimiter=",")
    for linha in leitor:
        dados.append(linha)

print(dados) 

# Carrega o CSV
df = pd.read_csv("Métricas RAG OpenAI - Página1.csv.csv")

# Substitui 'x' por NaN
df = df.replace("x", pd.NA)

# Conta quantos 'x' havia em cada coluna
x_counts = (df.isna().sum())  # porque transformamos 'x' em NaN

print("Número de valores 'x' (não contabilizados):")
print(x_counts)

# Converter colunas para numéricas (ignora NaN)
df["Context_relevance"] = pd.to_numeric(df["Context_relevance"], errors="coerce")
df["Faithfulness"] = pd.to_numeric(df["Faithfulness"], errors="coerce")

# Histograma das duas métricas
plt.hist(df["Context_relevance"].dropna(), bins=10, alpha=0.5, label="Context_relevance")
plt.hist(df["Faithfulness"].dropna(), bins=10, alpha=0.5, label="Faithfulness")
plt.xlabel("Valor")
plt.ylabel("Frequência")
plt.title("Distribuição das métricas")
plt.legend()
plt.show()

# Boxplot comparativo
df[["Context_relevance", "Faithfulness"]].plot(kind="box")
plt.title("Boxplot Context_relevance vs Faithfulness")
plt.show()





