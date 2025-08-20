from sqlalchemy import Column, String, Integer, FLOAT, TIMESTAMP, text, quoted_name
from db import Base

class RagResults(Base):
    __tablename__ = "Rag-Resultados"

    id = Column(Integer, primary_key=True, autoincrement=True)
    registrado_em = Column(quoted_name("Registrado em", True), TIMESTAMP(timezone=True), server_default=text("(now() AT TIME ZONE 'America/Sao_Paulo'::text)"))
    score1 = Column(quoted_name("Score 1", True), FLOAT)
    score2 = Column(quoted_name("Score 2", True), FLOAT)
    score3 = Column(quoted_name("Score 3", True), FLOAT)
    score4 = Column(quoted_name("Score 4", True), FLOAT)
    score5 = Column(quoted_name("Score 5", True), FLOAT)
    score6 = Column(quoted_name("Score 6", True), FLOAT)
    source = Column(quoted_name("Source", True), String)
    page = Column(quoted_name("Page", True), Integer)