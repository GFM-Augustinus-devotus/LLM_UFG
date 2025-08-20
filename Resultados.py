import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import pandas as pd

# Exemplo de dados: substitua por suas listas reais
# y_true: respostas corretas (0 = incorreta, 1 = correta)
# y_pred: respostas do chatbot (0 = incorreta, 1 = correta)
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

# Métricas
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
auc = roc_auc_score(y_true, y_pred)

print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"AUC-ROC: {auc:.2f}")

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap='Blues')
ax.set_title('Matriz de Confusão')
ax.set_xlabel('Previsto')
ax.set_ylabel('Real')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Negativo', 'Positivo'])
ax.set_yticklabels(['Negativo', 'Positivo'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('static/confusion_matrix.png')
plt.close()

# Curva ROC
fpr, tpr, _ = roc_curve(y_true, y_pred)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadeiro Positivo')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.savefig('static/roc_curve.png')
plt.close()

# Salvar métricas em CSV para exibir na aba HTML
df = pd.DataFrame({
    'Métrica': ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Valor': [accuracy, precision, recall, f1, auc]
})
df.to_csv('static/metricas.csv', index=False)