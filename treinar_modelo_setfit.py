# ============================================================
# TREINAMENTO DO MODELO — SETFIT (few-shot learning)
#
# SetFit é feito especificamente para classificação de texto
# com POUCOS exemplos rotulados (dezenas por classe, não
# milhares) — cenário mais próximo da sua base atual do que o
# fine-tuning completo de um BERT.
#
# Como funciona, resumidamente:
#   1) Parte de um sentence-transformer já pré-treinado (capta
#      bem a semântica de frases).
#   2) Faz um ajuste fino leve e contrastivo nesse encoder,
#      usando pares de exemplos — muito mais eficiente com
#      poucos dados do que treinar um classificador direto.
#   3) Treina uma cabeça de classificação simples (regressão
#      logística) em cima dos embeddings resultantes.
#
# Este é um dos 3 classificadores que você pode comparar:
#   - treinar_modelo.py          — BERTimbau
#   - treinar_modelo_setfit.py   (este arquivo) — SetFit
#   - treinar_modelo_tfidf.py    — TF-IDF + Regressão Logística
#
# IMPORTANTE: requer a biblioteca 'setfit' instalada
# (já incluída no requirements.txt). Se ainda não instalou:
#   pip install -r requirements.txt
# ============================================================

from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

from config import (
    DIR_MODELO_SETFIT,
    SETFIT_MODEL_NAME,
    SETFIT_EPOCHS,
    SETFIT_BATCH_SIZE,
    CLASSIFICATION_THRESHOLD,
    SEED,
)
from dados_treino import carregar_dados_treino


def linha():
    print("\n========================================")


# ============================================================
# 1. PREPARAR DADOS (compartilhado com BERT e TF-IDF)
# ============================================================
linha()
print("PREPARANDO DADOS")
linha()

d = carregar_dados_treino()

todas_tags = d["todas_tags"]
frequencia_tags = d["frequencia_tags"]
tags_removidas = d["tags_removidas"]
train_texts, test_texts = d["train_texts"], d["test_texts"]
train_labels, test_labels = d["train_labels"], d["test_labels"]

# ============================================================
# 2. MONTAR DATASETS NO FORMATO ESPERADO PELO SETFIT
#
# Para multi-label, o SetFit espera uma coluna "label" com uma
# lista de 0/1 por exemplo (mesma ordem de todas_tags).
# ============================================================
train_dataset = Dataset.from_dict(
    {"text": train_texts, "label": train_labels.tolist()}
)
test_dataset = Dataset.from_dict(
    {"text": test_texts, "label": test_labels.tolist()}
)

# ============================================================
# 3. CARREGAR MODELO BASE
#
# multi_target_strategy="one-vs-rest" treina um classificador
# binário por tag em cima do mesmo encoder — é a estratégia
# recomendada pelo SetFit para multi-label.
# ============================================================
linha()
print("CARREGANDO MODELO BASE DO SETFIT")
linha()
print("Modelo base:", SETFIT_MODEL_NAME)

model = SetFitModel.from_pretrained(
    SETFIT_MODEL_NAME,
    multi_target_strategy="one-vs-rest",
)

# ============================================================
# 4. TREINAR
# ============================================================
linha()
print("TREINANDO SETFIT")
linha()

args = TrainingArguments(
    batch_size=SETFIT_BATCH_SIZE,
    num_epochs=SETFIT_EPOCHS,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# ============================================================
# 5. AVALIAR
#
# Calculamos F1 macro/micro/weighted manualmente com o mesmo
# threshold usado pelo BERT (CLASSIFICATION_THRESHOLD), para
# a comparação entre os 3 modelos ser direta.
# ============================================================
linha()
print("AVALIAÇÃO — SETFIT")
linha()

raw_probs = model.predict_proba(test_texts)
probs = np.asarray(raw_probs)

# Defensivo: dependendo da versão do setfit, o retorno pode vir
# como (n_amostras, n_tags, 2) — probabilidade [negativa, positiva]
# por tag. Nesse caso, pegamos só a probabilidade da classe positiva.
if probs.ndim == 3:
    probs = probs[:, :, 1]

predictions = (probs >= CLASSIFICATION_THRESHOLD).astype(int)

avaliacao = {
    "eval_f1_macro": f1_score(test_labels, predictions, average="macro", zero_division=0),
    "eval_f1_micro": f1_score(test_labels, predictions, average="micro", zero_division=0),
    "eval_f1_weighted": f1_score(
        test_labels, predictions, average="weighted", zero_division=0
    ),
}
print(avaliacao)

# ============================================================
# 6. SALVAR MODELO
# ============================================================
Path(DIR_MODELO_SETFIT).mkdir(parents=True, exist_ok=True)

model.save_pretrained(DIR_MODELO_SETFIT)

with open(Path(DIR_MODELO_SETFIT) / "labels.txt", "w", encoding="utf-8") as f:
    for tag in todas_tags:
        f.write(tag + "\n")

frequencia_tags.to_csv(
    Path(DIR_MODELO_SETFIT) / "frequencia_tags.csv", index=False, encoding="utf-8"
)

if len(tags_removidas) > 0:
    tags_removidas.to_csv(
        Path(DIR_MODELO_SETFIT) / "tags_removidas_por_raridade.csv",
        index=False,
        encoding="utf-8",
    )

linha()
print("MODELO TREINADO E SALVO")
linha()
print("Modelo salvo em:", DIR_MODELO_SETFIT)
print(
    "\nCompare estas métricas de avaliação com as de "
    "treinar_modelo.py e treinar_modelo_tfidf.py, "
    "rodados sobre o MESMO conjunto de teste, para decidir "
    "qual classificador usar em produção."
)
