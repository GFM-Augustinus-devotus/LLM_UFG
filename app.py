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
    print("Recebido:", data)
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Pergunta não enviada"}), 400
    result = qa.invoke({"query": question})
    # Se o resultado for um dicionário, pegue o campo correto
    if isinstance(result, dict) and "result" in result:
        answer = result["result"]
    else:
        answer = str(result)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)