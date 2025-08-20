from db import Sessionlocal, engine, Base
from sqlalchemy import inspect, select
from rag_result import RagResults

Base.metadata.create_all(bind=engine)
insp = inspect(engine)
tabeelas = insp.get_table_names()

with Sessionlocal() as session:

    has_any = session.execute(select(RagResults.id).limit(1)).first() is not None

    #Fazer um ater table e colocar a coluna de perguntas para armazenar as perguntas