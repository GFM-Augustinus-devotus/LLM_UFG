# analise_csv.py
# -*- coding: utf-8 -*-
# Requisitos: pandas, numpy, matplotlib

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_CSV = "Métricas RAG OpenAI - Página1.csv"

def read_csv_robust(path: str) -> pd.DataFrame:
    """
    Leitura robusta:
    - Detecta separador automaticamente (engine='python')
    - Tenta utf-8; se falhar, tenta latin1
    - Trata 'x'/'X' como NA
    OBS: sem low_memory (não suportado pelo engine='python')
    """
    last_err = None
    for enc in ("utf-8", "latin1"):
        try:
            return pd.read_csv(
                path,
                sep=None,                 # autodetect
                engine="python",          # necessário para sep=None
                encoding=enc,
                na_values=["x", "X"],
                keep_default_na=True
            )
        except Exception as e:
            last_err = e
    raise last_err

def try_parse_numeric_series(s: pd.Series) -> pd.Series:
    """Converte série para numérico cobrindo vírgula decimal."""
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().sum() >= max(3, int(0.2 * len(s))):
        return s_num
    if s.dtype == object:
        s2 = s.str.replace(r"\.", "", regex=True).str.replace(",", ".", regex=False)
        s2_num = pd.to_numeric(s2, errors="coerce")
        if s2_num.notna().sum() > s_num.notna().sum():
            return s2_num
    return s_num

def detectar_eixo_x(df: pd.DataFrame):
    """Detecta coluna de eixo X (datas/tempo) ou usa o índice."""
    candidatos = [c for c in df.columns if c.lower() in {
        "date","data","datetime","tempo","time","dia","mês","mes","timestamp"
    }]
    for col in candidatos:
        x_dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        if x_dt.notna().sum() > max(3, int(0.2 * len(df))):
            return x_dt, col
        x_num = try_parse_numeric_series(df[col])
        if x_num.notna().sum() > max(3, int(0.2 * len(df))):
            return x_num, col
    if len(df.columns) > 0:
        first = df.columns[0]
        x_dt = pd.to_datetime(df[first], errors="coerce", dayfirst=True)
        if x_dt.notna().sum() > max(3, int(0.2 * len(df))):
            return x_dt, first
        x_num = try_parse_numeric_series(df[first])
        if x_num.notna().sum() > max(3, int(0.2 * len(df))):
            return x_num, first
    return pd.RangeIndex(start=0, stop=len(df), step=1), None

def preparar_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas potencialmente numéricas (respeitando vírgula decimal)."""
    df = df_raw.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].dtype == object:
            coerced = try_parse_numeric_series(df[col])
        else:
            coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() > 0:
            df[col] = coerced
    return df

def gerar_graficos(df_num: pd.DataFrame, x_vals, x_label: str | None, outdir: Path) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    plot_files = []
    for col in df_num.columns:
        serie = pd.to_numeric(df_num[col], errors="coerce")
        mask = pd.notna(serie) & pd.notna(x_vals)
        x_plot = x_vals[mask]
        y_plot = serie[mask]
        if y_plot.empty:
            continue
        plt.figure()
        plt.plot(x_plot, y_plot)  # sem cores definidas
        plt.title(f"Série: {col}")
        plt.xlabel(x_label if x_label else "Índice")
        plt.ylabel(col)
        plt.tight_layout()
        out_path = outdir / f"{col}_linha.png"
        plt.savefig(out_path.as_posix(), dpi=150)
        plt.close()
        plot_files.append(out_path.as_posix())
    return plot_files

def calcular_estatisticas(df_num: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df_num.columns:
        serie = pd.to_numeric(df_num[col], errors="coerce").dropna()
        if len(serie) == 0:
            continue
        rows.append({
            "coluna": col,
            "n": int(serie.count()),
            "media": float(serie.mean()),
            "variancia_amostral": float(serie.var(ddof=1)) if len(serie) > 1 else float("nan"),
            "desvio_padrao_amostral": float(serie.std(ddof=1)) if len(serie) > 1 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("coluna")

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.isfile(path):
        print(f"Arquivo não encontrado: {path}")
        sys.exit(1)

    df_raw = read_csv_robust(path)
    df = preparar_dataframe(df_raw)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("Nenhuma coluna numérica encontrada (após ignorar 'x'/'X').")
        sys.exit(0)

    x_vals, x_name = detectar_eixo_x(df_raw)

    plots_dir = Path("plots")
    gerar_graficos(df[numeric_cols], x_vals, x_name, plots_dir)

    stats_df = calcular_estatisticas(df[numeric_cols])
    stats_df.to_csv("Métricas RAG OpenAI - Página1.csv", index=False, encoding="utf-8")

    print("Concluído!")
    print(f"- Gráficos salvos em: {plots_dir.resolve()}")
    print(f"- Estatísticas salvas em: {Path('Métricas RAG OpenAI - Página1.csv').resolve()}")
