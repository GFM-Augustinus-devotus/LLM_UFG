const messages = document.getElementById('messages');
const questionInput = document.getElementById('question');
const sendBtn = document.getElementById('send');

function addMessage(text, sender) {
    const div = document.createElement('div');
    div.className = sender;
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

sendBtn.onclick = async function() {
    const question = questionInput.value.trim();
    if (!question) return;
    addMessage("Você: " + question, "user");
    questionInput.value = "";
    addMessage("Aguarde...", "bot");
    const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
    });
    const data = await response.json();
    messages.lastChild.textContent = "LLM: " + (data.answer || data.error);
};

questionInput.addEventListener("keydown", function(e) {
    if (e.key === "Enter") sendBtn.onclick();
});