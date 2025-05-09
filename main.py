#LLM-UFG ---> Large Language Model Aplicada na Organização de Resoluções Públicas

#----------------Bibliotecas--------------------------
from langchain.schema import StrOutputParser
from langchain_core.messages import SystemMessage , HumanMessage
from langchain_openai import ChatOpenAI
#----------------------------------------------------

with open(".gitignore" , "r") as arquivo:
    key = arquivo.read().strip()

print(key)

Mensagens = [
    SystemMessage("Traduza o texto para o português"),
    HumanMessage("Die Eisenfaust am Lanzenschaft")
]

modelo = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key = key)
parser = StrOutputParser()
chain = modelo | parser

resposta = chain.invoke(Mensagens)