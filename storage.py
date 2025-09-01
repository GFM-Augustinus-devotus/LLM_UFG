# storage.py
from __future__ import annotations
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import (
    create_engine, String, Float, Integer, DateTime, Text, JSON
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

# ==========================================
# Configuração da conexão
# - Use DATABASE_URL, ex:
#   postgresql+psycopg2://user:pass@host:5432/dbname
# - Ou defina as variáveis individuais abaixo.
# ==========================================
def _build_db_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    user = os.getenv("PGUSER", "postgres")
    pwd  = os.getenv("PGPASSWORD", "")
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    db   = os.getenv("PGDATABASE", "postgres")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


class Base(DeclarativeBase):
    pass


class RagRun(Base):
    """
    Uma linha por pergunta executada.
    Guarda pergunta, resposta (IA), melhor documento e métricas RAGAS.
    """
    __tablename__ = "rag_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Entrada/saída principal
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Melhor resultado do retriever
    top_source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    top_page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    top_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Métricas RAGAS
    context_relevance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    faithfulness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    answer_correctness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Extras úteis para auditoria
    all_scores: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)  # lista/dict c/ scores
    sources_used: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)  # [{source,page,score},...]


_engine = None
_Session = None


def init_db(echo: bool = False) -> None:
    """Cria engine, sessão e tabela (se não existir). Chame uma vez no início do programa."""
    global _engine, _Session
    if _engine is None:
        db_url = _build_db_url()
        _engine = create_engine(db_url, echo=echo, future=True)
        _Session = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
        Base.metadata.create_all(_engine)


def save_rag_run(
    *,
    question: str,
    answer: Optional[str],
    model_name: Optional[str],
    top_source: Optional[str],
    top_page: Optional[int],
    top_score: Optional[float],
    context_relevance: Optional[float],
    faithfulness: Optional[float],
    answer_correctness: Optional[float] = None,
    all_scores: Optional[List[Dict[str, Any]]] = None,
    sources_used: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Salva uma execução (pergunta) e retorna o id."""
    if _Session is None:
        raise RuntimeError("Chame init_db() antes de save_rag_run().")

    run = RagRun(
        question=question,
        answer=answer,
        model_name=model_name,
        top_source=top_source,
        top_page=top_page,
        top_score=top_score,
        context_relevance=context_relevance,
        faithfulness=faithfulness,
        answer_correctness=answer_correctness,
        all_scores=all_scores,
        sources_used=sources_used,
    )
    
    with _Session() as session:
        session.add(run)
        session.commit()
        session.refresh(run)
        return run.id
