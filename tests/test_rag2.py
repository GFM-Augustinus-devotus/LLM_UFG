import pytest

def test_analise_dez_perguntas_mesmo_documento():
    from rag2 import qa, get_trulens_recorder, Tru

    # Supondo que você já tem um documento carregado e pode gerar perguntas
    documento = "Seu texto base aqui..."
    perguntas = [f"O que diz o seguinte trecho? Parte {i}: {documento[:200]}" for i in range(10)]

    tru_recorder = get_trulens_recorder(qa, app_id="QA LLM v1 Teste")

    resultados = []
    with tru_recorder as recording:
        for pergunta in perguntas:
            resposta = qa.run(pergunta)
            resultados.append(resposta)

    assert len(resultados) == 10
    tru = Tru()
    leaderboard = tru.get_leaderboard()
    assert not leaderboard.empty