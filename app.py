from flask import Flask, request, jsonify, render_template
from llm import qa


app = Flask(__name__)

#Rotas do aplicativo

#rota para carregar o arquivo html da página
@app.route("/")
def home():
    return render_template("index.html")

#rota para carregar o local das perguntas
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    pergunta = data.get("pergunta", "")
    if not pergunta:
        return jsonify({"error": "Pergunta não enviada"}), 400
    resultado = qa.invoke({"query": pergunta})

    # Se o resultado for um dicionário, irá pegar o campo correto
    if isinstance(resultado, dict) and "resultado" in resultado:
        resposta = resultado["resultado"]
    else:
        resposta = str(resultado)
    return jsonify({"resposta": resposta})

if __name__ == "__main__":
    app.run(debug=True)