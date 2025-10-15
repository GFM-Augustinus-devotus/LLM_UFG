import pandas as pd
import matplotlib.pyplot as plt
import re
import unicodedata
from pathlib import Path

# === Configurações ===
FILE_PATH = "ragas_metrics - ragas_metrics.csv"   # ajuste se necessário
REQUESTED_COLS = [
    "answer_relevancy",
    "faithfulness",
    "context_relevane",   # typo comum
    "context_relevance",  # correto
    "answer_length",
    "Pontuação Final",
    "pontuacao final",    # fallback sem acento
]
SAVE_IMAGES = False  # True para salvar PNGs em ./plots

# === Utilitários ===
def normalize_text(s: str) -> str:
    """Normaliza texto: remove acentos, reduz espaços e baixa o case."""
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

# Regex com GRUPO DE CAPTURA para extrair o primeiro número da string
NUM_PATTERN = r"([-+]?\d*[.,]?\d+(?:[eE][-+]?\d+)?)"

def clean_numeric_series(s: pd.Series) -> pd.Series:
    """
    Converte uma série para float de forma robusta:
    - Troca vírgula decimal por ponto
    - Extrai o primeiro número da string (ex.: '65 com valor 1' -> 65)
    - Converte para float; valores inválidos viram NaN
    """
    as_str = s.astype(str).str.replace(",", ".", regex=False)
    extracted = as_str.str.extract(NUM_PATTERN, expand=False)
    return pd.to_numeric(extracted, errors="coerce")

def resolve_final_columns(df: pd.DataFrame, requested: list[str]) -> list[str]:
    """Resolve colunas existentes no CSV a partir de aliases/normalização."""
    norm_to_orig = {normalize_text(c): c for c in df.columns}
    final_cols, seen = [], set()
    for name in requested:
        norm = normalize_text(name)
        if norm in norm_to_orig:
            orig = norm_to_orig[norm]
            if orig not in seen:
                final_cols.append(orig)
                seen.add(orig)
    return final_cols

def ensure_output_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# === Pipeline principal ===
def main():
    # 1) Carregar CSV
    df = pd.read_csv(FILE_PATH)

    # 2) Resolver colunas-alvo
    final_cols = resolve_final_columns(df, REQUESTED_COLS)
    if not final_cols:
        print("Nenhuma das colunas solicitadas foi encontrada no CSV.")
        print("Colunas disponíveis:", list(df.columns))
        return

    print("Colunas detectadas no CSV:", list(df.columns))
    print("Colunas a processar:", final_cols)

    # 3) Limpeza numérica robusta
    for col in final_cols:
        df[col] = clean_numeric_series(df[col])

    # 4) (Opcional) Pasta para salvar imagens
    if SAVE_IMAGES:
        out_dir = ensure_output_dir("./plots")
        print(f"Salvando gráficos em: {out_dir.resolve()}")

    # 5) Plotar histogramas e boxplots
    plotted = []
    for col in final_cols:
        series = df[col].dropna()
        if series.empty:
            print(f"[AVISO] Coluna '{col}' não possui dados numéricos válidos após limpeza.")
            continue

        # Histograma
        plt.figure()
        series.plot(kind="hist", bins=20, title=f"Histograma - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequência")
        plt.tight_layout()
        if SAVE_IMAGES:
            (out_dir / f"{col}_hist.png").parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir / f"{col}_hist.png")
        plt.show()

        # Boxplot
        plt.figure()
        df.boxplot(column=[col])
        plt.title(f"Boxplot - {col}")
        plt.ylabel(col)
        plt.tight_layout()
        if SAVE_IMAGES:
            plt.savefig(out_dir / f"{col}_box.png")
        plt.show()

        plotted.append(col)

    # 6) Estatísticas descritivas
    stats_cols = [c for c in plotted if not df[c].dropna().empty]
    if stats_cols:
        print("\n=== Estatísticas descritivas ===")
        print(df[stats_cols].describe().T)
    else:
        print("\nNenhuma coluna com dados numéricos para mostrar estatísticas.")

if __name__ == "__main__":
    main()
