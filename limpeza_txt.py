PONTUACAO = (
    ".,;:!?\"'()[]{}-–—_/\\@%&+=*#ºª°§«»…$€"
)

PERMITIDOS = set(PT_LETRAS + DIGITOS + ESPACOS + PONTUACAO)

def limpar_texto(texto: str) -> str:
    # Normaliza para NFC (acentos combinados → forma composta)
    texto = unicodedata.normalize("NFC", texto)

    limpo = []
    for ch in texto:
        if ch in PERMITIDOS:
            limpo.append(ch)

    return "".join(limpo)

def main():
    ap = argparse.ArgumentParser(
        description="Remove caracteres não pertencentes ao português comum de um arquivo .txt e sobrescreve o arquivo."
    )
    ap.add_argument("arquivo", help="Caminho do arquivo .txt")
    ap.add_argument("--backup", action="store_true", help="Criar um .bak antes de sobrescrever")
    args = ap.parse_args()

    caminho = Path(args.arquivo)
    if not caminho.exists():
        raise SystemExit(f"Arquivo não encontrado: {caminho}")

    # Lê em UTF-8; caracteres inválidos serão ignorados
    original = caminho.read_text(encoding="utf-8", errors="ignore")

    limpo = limpar_texto(original)

    if args.backup:
        caminho.with_suffix(caminho.suffix + ".bak").write_text(original, encoding="utf-8")

    caminho.write_text(limpo, encoding="utf-8")
    print(f"Arquivo limpo e sobrescrito: {caminho}")