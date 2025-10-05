import pandas as pd
import matplotlib.pyplot as plt

ARQ = "MétricasRAG_Gemni.csv"

# Lê o CSV
df = pd.read_csv(
    ARQ,
    encoding="utf-8-sig",
    na_values=["x"],  # converte 'x' em NaN
)

# Limpa nomes de colunas
df.columns = [c.strip() for c in df.columns]

# Converte vírgula decimal para ponto
for col in ["Context_relevance", "Faithfulness"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove valores inválidos fora de [0,1] (como 500)
for col in ["Context_relevance", "Faithfulness"]:
    df.loc[(df[col] < 0) | (df[col] > 1), col] = pd.NA

# Estatísticas
stats = df[["Context_relevance", "Faithfulness"]].agg(
    ["mean", "var", "std"]
)
print("📊 Estatísticas das métricas:")
print(stats)

# Cria um índice para representar a sequência dos registros
df = df.reset_index().rename(columns={"index": "Registro"})

# Gráfico de linha das métricas
plt.figure(figsize=(12,6))
plt.plot(df["Registro"], df["Context_relevance"], marker="o", label="Context_relevance", alpha=0.7)
plt.plot(df["Registro"], df["Faithfulness"], marker="s", label="Faithfulness", alpha=0.7)
plt.xlabel("Registro")
plt.ylabel("Valor")
plt.title("Evolução das métricas por registro")
plt.legend()
plt.grid(True)
plt.show()

# Gráfico das médias como linha horizontal
plt.figure(figsize=(12,6))
for col, cor in zip(["Context_relevance", "Faithfulness"], ["blue", "orange"]):
    plt.plot(df["Registro"], df[col], alpha=0.6, label=col)
    plt.axhline(stats.loc["mean", col], color=cor, linestyle="--", label=f"Média {col}")
plt.xlabel("Registro")
plt.ylabel("Valor")
plt.title("Métricas com médias destacadas")
plt.legend()
plt.grid(True)
plt.show()





