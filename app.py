from flask import Flask, request, jsonify, render_template
from llm import load_qa

app = Flask(__name__)

# Só carrega o qa se não for o reloader do Flask
import os
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    _, qa = load_qa()
else:
    qa = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    print("Recebido:", data)
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Pergunta não enviada"}), 400
    if qa is None:
        return jsonify({"error": "QA não carregado"}), 500
    result = qa.invoke({"query": question})
    if isinstance(result, dict) and "result" in result:
        answer = result["result"]
    else:
        answer = str(result)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)