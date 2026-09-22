# ============================================================
# ANÁLISE DE UM NOVO RELATÓRIO
#
# Este script NÃO treina nada — ele carrega o modelo já
# treinado (salvo em DIR_MODELO por treinar_modelo.py) e usa
# esse modelo para analisar um relatório novo.
#
# Rode este script quantas vezes quiser, sempre que chegar um
# relatório novo. Só é preciso rodar treinar_modelo.py de novo
# se você adicionar novas quotations classificadas ao Excel de
# treino e quiser retreinar o modelo.
#
# Para trocar de relatório, edite NOVO_RELATORIO em config.py
# ============================================================

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (
    ARQUIVO_TREINO,
    NOVO_RELATORIO,
    DIR_MODELO,
    ARQUIVO_RESULTADO,
    MAX_LENGTH,
    BATCH_SIZE,
    MAX_CHARS_CHUNK,
    SIMILARITY_THRESHOLD,
    CLASSIFICATION_THRESHOLD,
    REVIEW_THRESHOLD,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", DEVICE)


def linha():
    print("\n========================================")


# ============================================================
# 0. VERIFICAR SE O MODELO JÁ FOI TREINADO
# ============================================================
labels_path = Path(DIR_MODELO) / "labels.txt"

if not labels_path.exists():
    raise FileNotFoundError(
        f"Não encontrei um modelo treinado em '{DIR_MODELO}'. "
        "Rode primeiro o script treinar_modelo.py."
    )

with open(labels_path, "r", encoding="utf-8") as f:
    todas_tags = [linha_.strip() for linha_ in f if linha_.strip()]

print("Tags carregadas do modelo:", len(todas_tags))

# ============================================================
# 1. LER AS QUOTATIONS CLASSIFICADAS
#
# Precisamos delas apenas para a ETAPA 1 (TF-IDF), que compara
# o relatório novo com as quotations humanas já conhecidas.
# O modelo BERT em si já está treinado e salvo — não é
# retreinado aqui.
# ============================================================
linha()
print("LENDO QUOTATIONS DE REFERÊNCIA (PARA TF-IDF)")
linha()

dados = pd.read_excel(ARQUIVO_TREINO)
dados["quotation"] = dados["quotation"].astype(str)

dados = dados[
    dados["quotation"].notna()
    & (dados["quotation"] != "")
    & (dados["quotation"] != "nan")
].reset_index(drop=True)

print("Quotations de referência:", len(dados))

# ============================================================
# 2. LER NOVO RELATÓRIO TXT
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
# 3. DIVIDIR RELATÓRIO EM PARÁGRAFOS
# ============================================================
paragrafos = re.split(r"\n\s*\n+", texto_relatorio)
paragrafos = [p.strip() for p in paragrafos]
paragrafos = [p for p in paragrafos if p != ""]

print("Parágrafos encontrados:", len(paragrafos))

# ============================================================
# 4. QUEBRAR PARÁGRAFOS MUITO GRANDES
# ============================================================
def quebrar_paragrafo(texto, max_chars=MAX_CHARS_CHUNK):
    if len(texto) <= max_chars:
        return [texto]

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
# 5. TF-IDF E SIMILARIDADE
# ============================================================
print("\nConstruindo representação TF-IDF...")

textos_referencia = dados["quotation"].tolist()

vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w{3,}\b",
)

vectorizer.fit(textos_referencia + trechos)

tfidf_referencia = vectorizer.transform(textos_referencia)
tfidf_novos = vectorizer.transform(trechos)

print("Termos no vocabulário:", len(vectorizer.vocabulary_))

similaridades = cosine_similarity(tfidf_novos, tfidf_referencia)

max_similarity = similaridades.max(axis=1)
indice_similar = similaridades.argmax(axis=1)
quotation_similar = dados["quotation"].iloc[indice_similar].values

# ============================================================
# 6. SELECIONAR CANDIDATOS
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
        "Tente diminuir SIMILARITY_THRESHOLD em config.py."
    )

# ============================================================
# 7. CARREGAR O MODELO JÁ TREINADO
# ============================================================
linha()
print("CARREGANDO MODELO TREINADO")
linha()

tokenizer_pred = AutoTokenizer.from_pretrained(DIR_MODELO)
model_pred = AutoModelForSequenceClassification.from_pretrained(DIR_MODELO)
model_pred.to(DEVICE)
model_pred.eval()


def classificar_trechos(textos, threshold=CLASSIFICATION_THRESHOLD):
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

    for j, tag in enumerate(todas_tags):
        nome_coluna = "prob_" + re.sub(r"[^0-9a-zA-Z_]", ".", tag)
        resultado[nome_coluna] = np.round(probs_all[:, j], 4)

    return resultado


# ============================================================
# 8. CLASSIFICAR OS TRECHOS
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
# 9. ORDENAR POR CONFIANÇA
# ============================================================
if len(resultados_finais) > 0:
    resultados_finais = resultados_finais.sort_values(
        "confianca", ascending=False
    ).reset_index(drop=True)

# ============================================================
# 10. FREQUÊNCIA DAS TAGS (referência, salva durante o treino)
# ============================================================
freq_path = Path(DIR_MODELO) / "frequencia_tags.csv"
if freq_path.exists():
    frequencia_tags = pd.read_csv(freq_path, encoding="utf-8")
else:
    frequencia_tags = pd.DataFrame({"tag": todas_tags})

# ============================================================
# 11. SALVAR RESULTADOS
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
# 12. RESUMO NA TELA
# ============================================================
if len(resultados_finais) > 0:
    print("\n\nRESUMO:\n")
    print(
        resultados_finais[
            ["trecho", "similaridade", "tags_preditas", "confianca", "revisar"]
        ].head(20)
    )
