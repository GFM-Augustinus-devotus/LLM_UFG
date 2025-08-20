from db import Sessionlocal, engine, Base
from sqlalchemy import inspect, select
from rag_result import RagResults

Base.metadata.create_all(bind=engine)
insp = inspect(engine)
tabeelas = insp.get_table_names()

with Sessionlocal() as session:

    has_any = session.execute(select(RagResults.id).limit(1)).first() is not None

    # if not has_any:
    #     session.add(
    #         pergunta = ""
    #         resposta = ""
    #         score1 = 0.0
    #         score2 = 0.0
    #         score3 = 0.0
    #         score4 = 0.0
    #         score5 = 0.0
    #         score6 = 0.0
    #         source = 0.0
    #         page = 12
    #     )
    #     session.commit()

    #Fazer um ater table e colocar a coluna de perguntas para armazenar as perguntas