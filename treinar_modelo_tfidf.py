# ============================================================
# TREINAMENTO DO MODELO — TF-IDF + REGRESSÃO LOGÍSTICA
#
# Baseline clássico: representa cada trecho por frequência de
# termos ponderada (TF-IDF) e treina um classificador linear
# simples (Regressão Logística) por tag.
#
# Vantagens deste modelo:
#   - Roda em segundos, em qualquer computador (sem GPU).
#   - Tem MUITO menos parâmetros que um BERT, então tende a
#     generalizar melhor com poucos dados (menos risco de
#     "decorar" em vez de aprender).
#   - Serve de referência: se um modelo mais sofisticado (BERT,
#     SetFit) não superar este baseline, não vale a pena usar o
#     modelo mais pesado.
#
# Este é um dos 3 classificadores que você pode comparar:
#   - treinar_modelo.py          — BERTimbau
#   - treinar_modelo_setfit.py   — SetFit
#   - treinar_modelo_tfidf.py    (este arquivo) — TF-IDF + LogReg
# ============================================================

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

from config import DIR_MODELO_TFIDF, CLASSIFICATION_THRESHOLD
from dados_treino import carregar_dados_treino


def linha():
    print("\n========================================")


# ============================================================
# 1. PREPARAR DADOS (compartilhado com BERT e SetFit)
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
# 2. VETORIZAÇÃO TF-IDF
#
# ngram_range=(1, 2) inclui também pares de palavras (bigramas),
# o que ajuda a capturar expressões (ex: "impacto social") que
# uma palavra isolada não capta.
# ============================================================
linha()
print("VETORIZANDO COM TF-IDF")
linha()

vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w{3,}\b",
    ngram_range=(1, 2),
    max_features=20000,
)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

print("Termos no vocabulário:", len(vectorizer.vocabulary_))

# ============================================================
# 3. TREINAR UM CLASSIFICADOR LINEAR POR TAG
#
# class_weight="balanced" já compensa automaticamente o
# desbalanceamento entre tags frequentes e raras — o
# equivalente ao pos_weight usado no script do BERT.
# ============================================================
linha()
print("TREINANDO REGRESSÃO LOGÍSTICA (uma por tag)")
linha()

base_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf = OneVsRestClassifier(base_clf)
clf.fit(X_train, train_labels)

# ============================================================
# 4. AVALIAR
# ============================================================
linha()
print("AVALIAÇÃO — TF-IDF + REGRESSÃO LOGÍSTICA")
linha()

probs = clf.predict_proba(X_test)
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
# 5. SALVAR MODELO
# ============================================================
Path(DIR_MODELO_TFIDF).mkdir(parents=True, exist_ok=True)

joblib.dump(vectorizer, Path(DIR_MODELO_TFIDF) / "vectorizer.joblib")
joblib.dump(clf, Path(DIR_MODELO_TFIDF) / "classificador.joblib")

with open(Path(DIR_MODELO_TFIDF) / "labels.txt", "w", encoding="utf-8") as f:
    for tag in todas_tags:
        f.write(tag + "\n")

frequencia_tags.to_csv(
    Path(DIR_MODELO_TFIDF) / "frequencia_tags.csv", index=False, encoding="utf-8"
)

if len(tags_removidas) > 0:
    tags_removidas.to_csv(
        Path(DIR_MODELO_TFIDF) / "tags_removidas_por_raridade.csv",
        index=False,
        encoding="utf-8",
    )

linha()
print("MODELO TREINADO E SALVO")
linha()
print("Modelo salvo em:", DIR_MODELO_TFIDF)
print(
    "\nCompare estas métricas de avaliação com as de "
    "treinar_modelo.py e treinar_modelo_setfit.py, "
    "rodados sobre o MESMO conjunto de teste, para decidir "
    "qual classificador usar em produção."
)
