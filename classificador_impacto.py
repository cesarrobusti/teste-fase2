# ============================================================
# CLASSIFICADOR AUTOMÁTICO DE IMPACTO DE PESQUISA
#
# ETAPA 1:
#   Localização de trechos potencialmente relevantes
#   usando similaridade TF-IDF com quotations anotadas.
#
# ETAPA 2:
#   Classificação multi-label usando BERTimbau.
#
# ENTRADA:
#   Excel com quotations classificadas
#   +
#   arquivo TXT de um novo relatório
#
# SAÍDA:
#   Excel contendo:
#       - trecho
#       - tags previstas
#       - probabilidade de cada tag
#       - confiança
#       - indicação de revisão humana
#
# ============================================================
#
# COMO USAR (VSCode):
#
# 1) Instale o Python 3.10, 3.11 ou 3.12 (evite a versão mais
#    recente lançada há poucas semanas; torch demora a dar
#    suporte a versões novas).
#
# 2) Crie um ambiente virtual dentro da pasta do projeto:
#
#       python -m venv .venv
#
# 3) Ative o ambiente:
#
#       Windows (PowerShell):  .venv\\Scripts\\Activate.ps1
#       Windows (cmd):         .venv\\Scripts\\activate.bat
#
# 4) Instale as dependências (arquivo requirements.txt
#    fornecido junto com este script):
#
#       pip install -r requirements.txt
#
# 5) No VSCode, selecione o interpretador do .venv
#    (Ctrl+Shift+P -> "Python: Select Interpreter").
#
# 6) Ajuste as variáveis na seção "CONFIGURAÇÃO" abaixo e
#    rode o script (F5 ou "python classificador_impacto.py").
#
# ============================================================

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ============================================================
# 0. CONFIGURAÇÃO
# ============================================================

# ------------------------------------------------------------
# PASTA DE TRABALHO
# ------------------------------------------------------------
DIR_TRABALHO = r"C:\Users\cesar\Downloads\Nova pasta"
os.chdir(DIR_TRABALHO)

# ------------------------------------------------------------
# ARQUIVO COM AS QUOTATIONS JÁ CLASSIFICADAS
# ------------------------------------------------------------
ARQUIVO_TREINO = "Artigo - Impact Tagging-quotations.xlsx"

# ------------------------------------------------------------
# NOVO RELATÓRIO
#
# Basta trocar este arquivo quando quiser analisar outro.
# ------------------------------------------------------------
NOVO_RELATORIO = "Proposta_24009016_UNIVERSIDADE_FEDERAL_DE_CAMPINA_GRANDE_(UFCG).txt"

# ------------------------------------------------------------
# ARQUIVOS / DIRETÓRIOS
# ------------------------------------------------------------
DIR_MODELO = "./modelo_impacto"
ARQUIVO_RESULTADO = "resultado_relatorio_159.xlsx"

# ------------------------------------------------------------
# MODELO EM PORTUGUÊS
# ------------------------------------------------------------
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

# ------------------------------------------------------------
# CONFIGURAÇÕES DO BERT
# ------------------------------------------------------------
MAX_LENGTH = 256
EPOCHS = 4
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
SEED = 42

# ------------------------------------------------------------
# CONFIGURAÇÕES DA ETAPA 1
# ------------------------------------------------------------
# Tamanho aproximado dos trechos analisados.
# O código tenta preservar parágrafos.
# Se os parágrafos forem muito grandes, eles serão quebrados.
MAX_CHARS_CHUNK = 1200

# ------------------------------------------------------------
# Similaridade mínima para considerar um trecho candidato.
#
# 0.10 = bastante inclusivo
# 0.20 = intermediário
# 0.30 = mais restritivo
#
# Recomendo começar com 0.15.
# ------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.15

# ------------------------------------------------------------
# THRESHOLD DO BERT
#
# Uma tag será retornada quando sua probabilidade for >= 0.50.
# ------------------------------------------------------------
CLASSIFICATION_THRESHOLD = 0.50

# ------------------------------------------------------------
# CONFIANÇA PARA REVISÃO HUMANA
#
# Se a maior probabilidade de qualquer tag for inferior a
# este valor, o trecho será marcado para revisão.
# ------------------------------------------------------------
REVIEW_THRESHOLD = 0.70

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linha():
    print("\n========================================")


# ============================================================
# 1. LER AS QUOTATIONS CLASSIFICADAS
# ============================================================
linha()
print("LENDO DADOS CLASSIFICADOS")
linha()

dados = pd.read_excel(ARQUIVO_TREINO)

print("\nColunas encontradas:")
print(list(dados.columns))

if "quotation" not in dados.columns:
    raise ValueError("A coluna 'quotation' não foi encontrada.")
if "codes" not in dados.columns:
    raise ValueError("A coluna 'codes' não foi encontrada.")

# ============================================================
# 2. LIMPEZA
# ============================================================
dados["quotation"] = dados["quotation"].astype(str)
dados["codes"] = dados["codes"].astype(str)

dados = dados[
    dados["quotation"].notna()
    & (dados["quotation"] != "")
    & (dados["quotation"] != "nan")
    & dados["codes"].notna()
    & (dados["codes"] != "")
    & (dados["codes"] != "nan")
].reset_index(drop=True)

print("\nQuotations classificadas:", len(dados))

# ============================================================
# 3. TRANSFORMAR CODES EM TAGS
# ============================================================
def separar_tags(x):
    x = str(x)
    tags = re.split(r"\s*[,;|]\s*", x)
    tags = [t.strip() for t in tags if t and t.strip()]
    # remove duplicatas preservando ordem
    vistos = set()
    resultado = []
    for t in tags:
        if t not in vistos:
            vistos.add(t)
            resultado.append(t)
    return resultado


dados["tags"] = dados["codes"].apply(separar_tags)

# ============================================================
# 4. LISTA DE TODAS AS TAGS
# ============================================================
todas_tags = sorted(set(tag for tags in dados["tags"] for tag in tags))

print("\nNúmero de tags:", len(todas_tags))
print(todas_tags)

# ============================================================
# 5. FREQUÊNCIA DAS TAGS
# ============================================================
frequencia_tags = pd.DataFrame(
    {
        "tag": todas_tags,
        "n": [sum(tag in tags for tags in dados["tags"]) for tag in todas_tags],
    }
).sort_values("n", ascending=False).reset_index(drop=True)

print("\nFrequência das tags:")
print(frequencia_tags)

# ============================================================
# 6. MATRIZ MULTI-LABEL
# ============================================================
label_matrix = np.zeros((len(dados), len(todas_tags)), dtype=np.int64)
tag_to_idx = {tag: i for i, tag in enumerate(todas_tags)}

for i, tags in enumerate(dados["tags"]):
    for tag in tags:
        if tag in tag_to_idx:
            label_matrix[i, tag_to_idx[tag]] = 1

# ============================================================
# 7. DIVISÃO TRAIN / TEST
# ============================================================
indices = np.arange(len(dados))
train_idx, test_idx = train_test_split(
    indices, test_size=0.20, random_state=SEED
)

train_texts = dados["quotation"].iloc[train_idx].tolist()
test_texts = dados["quotation"].iloc[test_idx].tolist()
train_labels = label_matrix[train_idx]
test_labels = label_matrix[test_idx]

print("\nTreinamento:", len(train_texts))
print("Teste:", len(test_texts))

# ============================================================
# 8. TOKENIZER
# ============================================================
linha()
print("CARREGANDO BERTIMBAU")
linha()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================================================
# 9. TOKENIZAÇÃO
# ============================================================
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
# 10. DATASET PYTORCH
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
# 11. MODELO BERT MULTI-LABEL
# ============================================================
num_labels = len(todas_tags)
print("\nNúmero de labels:", num_labels)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification",
)

# ============================================================
# 12. MÉTRICAS
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs >= 0.50).astype(int)

    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)

    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
    }


# ============================================================
# 13. ARGUMENTOS DO TREINAMENTO
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
# 14. TRAINER
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ============================================================
# 15. TREINAR
# ============================================================
linha()
print("TREINANDO BERTIMBAU")
linha()

trainer.train()

# ============================================================
# 16. AVALIAR
# ============================================================
linha()
print("AVALIAÇÃO")
linha()

avaliacao = trainer.evaluate()
print(avaliacao)

# ============================================================
# 17. SALVAR MODELO
# ============================================================
Path(DIR_MODELO).mkdir(parents=True, exist_ok=True)

trainer.save_model(DIR_MODELO)
tokenizer.save_pretrained(DIR_MODELO)

# equivalente ao saveRDS: salvamos a lista de tags em texto simples
with open(os.path.join(DIR_MODELO, "labels.txt"), "w", encoding="utf-8") as f:
    for tag in todas_tags:
        f.write(tag + "\n")

print("\nModelo salvo em:", DIR_MODELO)

# ============================================================
# ============================================================
#
# ETAPA 1
# LOCALIZAÇÃO DE TRECHOS POTENCIALMENTE RELEVANTES
#
# ============================================================
# ============================================================

# ============================================================
# 18. LER NOVO RELATÓRIO TXT
# ============================================================
linha()
print("LENDO NOVO RELATÓRIO")
linha()

with open(NOVO_RELATORIO, "r", encoding="utf-8") as f:
    texto_relatorio = f.read()

if len(texto_relatorio.strip()) == 0:
    raise ValueError("O relatório está vazio.")

print("Caracteres:", len(texto_relatorio))

# ============================================================
# 19. DIVIDIR RELATÓRIO EM PARÁGRAFOS
# ============================================================
paragrafos = re.split(r"\n\s*\n+", texto_relatorio)
paragrafos = [p.strip() for p in paragrafos]
paragrafos = [p for p in paragrafos if p != ""]

print("Parágrafos encontrados:", len(paragrafos))

# ============================================================
# 20. QUEBRAR PARÁGRAFOS MUITO GRANDES
# ============================================================
def quebrar_paragrafo(texto, max_chars=MAX_CHARS_CHUNK):
    if len(texto) <= max_chars:
        return [texto]

    # Tenta primeiro separar por frases
    frases = re.split(r"(?<=[.!?])\s+", texto)

    resultado = []
    atual = ""

    for frase in frases:
        candidato = (atual + " " + frase).strip()

        if len(candidato) <= max_chars:
            atual = candidato
        else:
            if len(atual) > 0:
                resultado.append(atual)
            atual = frase

    if len(atual) > 0:
        resultado.append(atual)

    return resultado


trechos = []
for p in paragrafos:
    trechos.extend(quebrar_paragrafo(p))

trechos = [t.strip() for t in trechos]
trechos = [t for t in trechos if len(t) >= 40]

print("Trechos para análise:", len(trechos))

# ============================================================
# 21. TF-IDF E SIMILARIDADE
# ============================================================
# A etapa 1 compara cada trecho novo com as quotations
# humanas que já conhecemos.
#
# Não é ainda um classificador supervisionado de evidência.
# É um filtro de candidatos.
#
# Usamos o TfidfVectorizer do scikit-learn (equivalente ao
# TF-IDF manual construído no script original em R).
print("\nConstruindo representação TF-IDF...")

textos_referencia = dados["quotation"].tolist()

vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w{3,}\b",  # tokens com 3+ caracteres, como no original
)

# Vocabulário construído a partir de referência + trechos novos
vectorizer.fit(textos_referencia + trechos)

tfidf_referencia = vectorizer.transform(textos_referencia)
tfidf_novos = vectorizer.transform(trechos)

print("Termos no vocabulário:", len(vectorizer.vocabulary_))

# ============================================================
# 22. SIMILARIDADE
# ============================================================
similaridades = cosine_similarity(tfidf_novos, tfidf_referencia)

max_similarity = similaridades.max(axis=1)
indice_similar = similaridades.argmax(axis=1)
quotation_similar = dados["quotation"].iloc[indice_similar].values

# ============================================================
# 23. SELECIONAR CANDIDATOS
# ============================================================
candidatos = pd.DataFrame(
    {
        "trecho_id": range(1, len(trechos) + 1),
        "trecho": trechos,
        "similaridade": np.round(max_similarity, 4),
        "quotation_referencia": quotation_similar,
    }
)

candidatos = candidatos[candidatos["similaridade"] >= SIMILARITY_THRESHOLD].reset_index(
    drop=True
)

print("\nTrechos considerados candidatos:", len(candidatos))

if len(candidatos) == 0:
    warnings.warn(
        f"Nenhum trecho ultrapassou o threshold de {SIMILARITY_THRESHOLD}. "
        "Tente diminuir SIMILARITY_THRESHOLD."
    )

# ============================================================
# ============================================================
#
# ETAPA 2
# CLASSIFICAÇÃO BERT
#
# ============================================================
# ============================================================

# ============================================================
# 24. FUNÇÃO PARA CLASSIFICAR CANDIDATOS
# ============================================================
def classificar_trechos(textos, threshold=CLASSIFICATION_THRESHOLD):
    tokenizer_pred = AutoTokenizer.from_pretrained(DIR_MODELO)
    model_pred = AutoModelForSequenceClassification.from_pretrained(DIR_MODELO)
    model_pred.to(DEVICE)
    model_pred.eval()

    resultados = []

    with torch.no_grad():
        for inicio in range(0, len(textos), BATCH_SIZE):
            fim = min(inicio + BATCH_SIZE, len(textos))
            batch = textos[inicio:fim]

            enc = tokenizer_pred(
                batch,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            output = model_pred(**enc)
            logits = output.logits.detach().cpu().numpy()

            probs = 1 / (1 + np.exp(-logits))
            resultados.append(probs)

    probs_all = np.vstack(resultados)

    # Tags previstas
    tags_preditas = []
    for x in probs_all:
        selecionadas = [todas_tags[i] for i in range(len(todas_tags)) if x[i] >= threshold]
        tags_preditas.append(" | ".join(selecionadas) if selecionadas else None)

    confianca = probs_all.max(axis=1)
    numero_tags = (probs_all >= threshold).sum(axis=1)

    resultado = pd.DataFrame(
        {
            "trecho": textos,
            "tags_preditas": tags_preditas,
            "confianca": np.round(confianca, 4),
            "numero_tags": numero_tags,
            "revisar": confianca < REVIEW_THRESHOLD,
        }
    )

    # Probabilidade de cada tag
    for j, tag in enumerate(todas_tags):
        nome_coluna = "prob_" + re.sub(r"[^0-9a-zA-Z_]", ".", tag)
        resultado[nome_coluna] = np.round(probs_all[:, j], 4)

    return resultado


# ============================================================
# 25. CLASSIFICAR OS TRECHOS
# ============================================================
if len(candidatos) > 0:
    linha()
    print("CLASSIFICANDO COM BERTIMBAU")
    linha()

    classificacoes = classificar_trechos(
        candidatos["trecho"].tolist(), CLASSIFICATION_THRESHOLD
    )

    resultados_finais = pd.concat(
        [candidatos.reset_index(drop=True), classificacoes.drop(columns=["trecho"])],
        axis=1,
    )
else:
    resultados_finais = candidatos.copy()

# ============================================================
# 26. ORDENAR POR CONFIANÇA
# ============================================================
if len(resultados_finais) > 0:
    resultados_finais = resultados_finais.sort_values(
        "confianca", ascending=False
    ).reset_index(drop=True)

# ============================================================
# 27. SALVAR RESULTADOS
# ============================================================
print("\nSalvando resultados...")

with pd.ExcelWriter(ARQUIVO_RESULTADO, engine="openpyxl") as writer:
    resultados_finais.to_excel(writer, sheet_name="resultados", index=False)
    frequencia_tags.to_excel(writer, sheet_name="frequencia_tags", index=False)

linha()
print("CONCLUÍDO")
linha()

print("Arquivo:", ARQUIVO_RESULTADO)
print("Trechos analisados:", len(resultados_finais))

if len(resultados_finais) > 0:
    print("Trechos para revisão:", int(resultados_finais["revisar"].sum()))

# ============================================================
# 28. RESUMO NA TELA
# ============================================================
if len(resultados_finais) > 0:
    print("\n\nRESUMO:\n")
    print(
        resultados_finais[
            ["trecho", "similaridade", "tags_preditas", "confianca", "revisar"]
        ].head(20)
    )
