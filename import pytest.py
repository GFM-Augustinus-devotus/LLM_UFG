import pytest
from unittest.mock import MagicMock, patch
from rag2 import get_trulens_recorder, metrics

# File: tests/test_rag2.py

def test_get_trulens_recorder_returns_truchain():
    mock_qa = MagicMock()
    app_id = "test_app"
    with patch("rag2.TruChain") as MockTruChain:
        recorder = get_trulens_recorder(mock_qa, app_id)
        MockTruChain.assert_called_once_with(app=mock_qa, feedbacks=metrics, app_id=app_id)
        assert recorder == MockTruChain.return_value

def test_metrics_contains_feedbacks():
    names = [f.name for f in metrics]
    assert "Answer Relevance" in names
    assert "Context Relevance" in names
    assert "Groundedness" in names
    assert "Answer Correctness" in names

@patch("rag2.get_trulens_recorder")
@patch("rag2.qa")
def test_run_multiple_questions(mock_qa, mock_get_recorder):
    # Setup
    recorder = MagicMock()
    mock_get_recorder.return_value.__enter__.return_value = recorder
    mock_qa.run.side_effect = [f"Resposta {i}" for i in range(10)]
    perguntas = [f"Pergunta {i}" for i in range(10)]

    # Simulate loop for 10 questions
    for pergunta in perguntas:
        with get_trulens_recorder(mock_qa, "QA LLM v1") as recording:
            resposta = mock_qa.run(pergunta)
            assert resposta.startswith("Resposta")

    assert mock_qa.run.call_count == 10