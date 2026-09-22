# ============================================================
# TREINAMENTO DO MODELO — BERTIMBAU (fine-tuning completo)
#
# Rode este script apenas quando quiser (re)treinar o modelo,
# por exemplo:
#   - na primeira vez que for usar o classificador
#   - depois de adicionar novas quotations classificadas ao
#     Excel de treino
#
# Ao final, o modelo treinado fica salvo em DIR_MODELO
# (definido em config.py). Depois disso, use
# analisar_relatorio.py (ou app.py) para analisar quantos
# relatórios quiser, SEM precisar rodar este script de novo.
#
# Este é um dos 3 classificadores que você pode comparar:
#   - treinar_modelo.py          (este arquivo) — BERTimbau
#   - treinar_modelo_setfit.py   — SetFit (few-shot)
#   - treinar_modelo_tfidf.py    — TF-IDF + Regressão Logística
# ============================================================

from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from config import (
    DIR_MODELO,
    MODEL_NAME,
    MAX_LENGTH,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    SEED,
    POS_WEIGHT_CAP,
)
from dados_treino import carregar_dados_treino

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", DEVICE)


def linha():
    print("\n========================================")


# ============================================================
# 1. PREPARAR DADOS (compartilhado com SetFit e TF-IDF)
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
# 2. TOKENIZER
# ============================================================
linha()
print("CARREGANDO BERTIMBAU")
linha()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenizar(textos):
    return tokenizer(
        textos,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )


train_encodings = tokenizar(train_texts)
test_encodings = tokenizar(test_texts)

# ============================================================
# 3. DATASET PYTORCH
# ============================================================
class ImpactDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx], dtype=torch.long)
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = ImpactDataset(train_encodings, train_labels.astype(np.float32))
test_dataset = ImpactDataset(test_encodings, test_labels.astype(np.float32))

# ============================================================
# 4. MODELO BERT MULTI-LABEL
# ============================================================
num_labels = len(todas_tags)
print("\nNúmero de labels:", num_labels)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification",
)

# ============================================================
# 5. PESO DAS CLASSES POSITIVAS (pos_weight)
#
# Em problemas multi-label esparsos, o modelo tende a "chutar"
# probabilidade baixa para tudo, pois isso já minimiza bem o
# erro médio. O pos_weight penaliza mais os erros em exemplos
# positivos das tags mais raras, forçando o modelo a arriscar
# mais em vez de sempre prever perto de zero.
# ============================================================
contagem_positivos = train_labels.sum(axis=0)
contagem_negativos = train_labels.shape[0] - contagem_positivos

pos_weight_valores = np.clip(
    contagem_negativos / np.maximum(contagem_positivos, 1),
    a_min=1.0,
    a_max=POS_WEIGHT_CAP,
)
pos_weight_tensor = torch.tensor(pos_weight_valores, dtype=torch.float)

print("\nPesos aplicados às classes positivas (pos_weight):")
for tag, peso in zip(todas_tags, pos_weight_valores):
    print(f"  {tag}: {peso:.2f}")


class WeightedTrainer(Trainer):
    """Trainer que usa BCEWithLogitsLoss com pos_weight, em vez da
    perda padrão sem ponderação usada pelo Trainer genérico."""

    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device)
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ============================================================
# 6. MÉTRICAS
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs >= 0.50).astype(int)

    return {
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
        "f1_weighted": f1_score(
            labels, predictions, average="weighted", zero_division=0
        ),
    }


# ============================================================
# 7. ARGUMENTOS DO TREINAMENTO
# ============================================================
training_args = TrainingArguments(
    output_dir=DIR_MODELO,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.10,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
)

# ============================================================
# 8. TRAINER
# ============================================================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    pos_weight=pos_weight_tensor,
)

# ============================================================
# 9. TREINAR
# ============================================================
linha()
print("TREINANDO BERTIMBAU")
linha()

trainer.train()

# ============================================================
# 10. AVALIAR
# ============================================================
linha()
print("AVALIAÇÃO — BERTIMBAU")
linha()

avaliacao = trainer.evaluate()
print(avaliacao)

# ============================================================
# 11. SALVAR MODELO
# ============================================================
Path(DIR_MODELO).mkdir(parents=True, exist_ok=True)

trainer.save_model(DIR_MODELO)
tokenizer.save_pretrained(DIR_MODELO)

with open(Path(DIR_MODELO) / "labels.txt", "w", encoding="utf-8") as f:
    for tag in todas_tags:
        f.write(tag + "\n")

frequencia_tags.to_csv(
    Path(DIR_MODELO) / "frequencia_tags.csv", index=False, encoding="utf-8"
)

if len(tags_removidas) > 0:
    tags_removidas.to_csv(
        Path(DIR_MODELO) / "tags_removidas_por_raridade.csv",
        index=False,
        encoding="utf-8",
    )

linha()
print("MODELO TREINADO E SALVO")
linha()
print("Modelo salvo em:", DIR_MODELO)
print(
    "\nCompare estas métricas de avaliação com as de "
    "treinar_modelo_setfit.py e treinar_modelo_tfidf.py, "
    "rodados sobre o MESMO conjunto de teste, para decidir "
    "qual classificador usar em produção."
)
