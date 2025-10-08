import pandas as pd
import matplotlib.pyplot as plt

# === Carregar CSV ===
file_path = "ragas_metrics - Atualizada.csv" 
df = pd.read_csv(file_path)

# Mapeamento para corrigir variações de nomes
name_map = {
    "answer_relevancy": "answer_relevancy",
    "faithfulness": "faithfulness",
    "context_relevane": "context_relevance",  # typo corrigido
    "answer_length": "answer_length",   # sinônimo comum
}

requested_cols = ["answer_relevancy", "faithfulness", "context_relevane", "answer_length"]

# Resolver colunas finais a plotar (apenas as que existem)
final_cols = []
for c in requested_cols:
    target = name_map.get(c, c)
    if target in df.columns:
        final_cols.append(target)

# Converter vírgula decimal para ponto e para float
for col in final_cols:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

# === Plotar: 1 histograma + 1 boxplot para cada coluna ===
for col in final_cols:
    # Histograma
    plt.figure()
    df[col].plot(kind="hist", bins=20, title=f"Histograma - {col}")
    plt.xlabel(col)
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.show()

    # Boxplot
    plt.figure()
    df.boxplot(column=[col])
    plt.title(f"Boxplot - {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

print("Colunas plotadas:", final_cols)
