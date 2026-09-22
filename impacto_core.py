# ============================================================
# LÓGICA COMPARTILHADA DE PROCESSAMENTO
#
# Este módulo contém as mesmas funções usadas em
# analisar_relatorio.py, reorganizadas como funções reutilizáveis
# para que o app.py (interface Streamlit) possa chamá-las.
#
# Não precisa rodar este arquivo diretamente.
# ============================================================

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from docx import Document

from config import (
    ARQUIVO_TREINO,
    DIR_MODELO,
    MAX_LENGTH,
    BATCH_SIZE,
    MAX_CHARS_CHUNK,
    SIMILARITY_THRESHOLD,
    CLASSIFICATION_THRESHOLD,
    REVIEW_THRESHOLD,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def modelo_disponivel():
    """Verifica se o modelo está acessível localmente ou no Hugging Face."""
    # Como o modelo está no Hugging Face (ou numa pasta local válida), retorna True
    return True


def carregar_tags():
    # Tenta achar localmente; se não achar, baixa do Hugging Face
    labels_local = Path("labels.txt")
    if labels_local.exists():
        caminho = labels_local
    else:
        caminho = hf_hub_download(repo_id=DIR_MODELO, filename="labels.txt")
        
    with open(caminho, "r", encoding="utf-8") as f:
        return [linha.strip() for linha in f if linha.strip()]


def carregar_referencia():
    """Carrega as quotations humanas usadas na Etapa 1 (TF-IDF)."""
    dados = pd.read_excel(ARQUIVO_TREINO)
    dados["quotation"] = dados["quotation"].astype(str)
    dados = dados[
        dados["quotation"].notna()
        & (dados["quotation"] != "")
        & (dados["quotation"] != "nan")
    ].reset_index(drop=True)
    return dados


def carregar_modelo():
    """Carrega o modelo BERT diretamente do Hugging Face (ou pasta local)."""
    tokenizer = AutoTokenizer.from_pretrained(DIR_MODELO)
    model = AutoModelForSequenceClassification.from_pretrained(DIR_MODELO)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def carregar_frequencia_tags():
    freq_local = Path("frequencia_tags.csv")
    if freq_local.exists():
        return pd.read_csv(freq_local, encoding="utf-8")
    
    try:
        caminho = hf_hub_download(repo_id=DIR_MODELO, filename="frequencia_tags.csv")
        return pd.read_csv(caminho, encoding="utf-8")
    except Exception:
        # Se não houver o arquivo CSV, gera a partir das tags
        return pd.DataFrame({"tag": carregar_tags()})


def carregar_frequencia_tags():
    freq_path = Path(DIR_MODELO) / "frequencia_tags.csv"
    if freq_path.exists():
        return pd.read_csv(freq_path, encoding="utf-8")
    return pd.DataFrame({"tag": carregar_tags()})


EXTENSOES_SUPORTADAS = ("txt", "docx")


def extrair_texto_txt(conteudo_bytes):
    return conteudo_bytes.decode("utf-8", errors="replace")


def extrair_texto_docx(conteudo_bytes):
    documento = Document(io.BytesIO(conteudo_bytes))
    paragrafos = [p.text.strip() for p in documento.paragraphs]
    paragrafos = [p for p in paragrafos if p]
    # Junta com linha em branco entre parágrafos, para preservar a
    # mesma lógica de divisão em trechos usada para .txt.
    return "\n\n".join(paragrafos)


def extrair_texto(nome_arquivo, conteudo_bytes):
    """Extrai o texto de um arquivo enviado, de acordo com a extensão.

    Suporta .txt e .docx. Arquivos .doc (formato antigo do Word,
    anterior a 2007) não são suportados diretamente — é preciso
    converter para .docx primeiro.
    """
    ext = nome_arquivo.lower().rsplit(".", 1)[-1] if "." in nome_arquivo else ""

    if ext == "txt":
        return extrair_texto_txt(conteudo_bytes)
    elif ext == "docx":
        return extrair_texto_docx(conteudo_bytes)
    elif ext == "doc":
        raise ValueError(
            "Arquivos .doc (formato antigo do Word) não são suportados. "
            "Abra o arquivo no Word e use 'Salvar como' > 'Word Document (.docx)', "
            "depois envie o arquivo .docx gerado."
        )
    else:
        raise ValueError(
            f"Formato '.{ext}' não suportado. Envie um arquivo .txt ou .docx."
        )


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


def dividir_em_trechos(texto_relatorio):
    """Divide o texto do relatório em parágrafos/trechos analisáveis."""
    paragrafos = re.split(r"\n\s*\n+", texto_relatorio)
    paragrafos = [p.strip() for p in paragrafos]
    paragrafos = [p for p in paragrafos if p != ""]

    trechos = []
    for p in paragrafos:
        trechos.extend(quebrar_paragrafo(p))

    trechos = [t.strip() for t in trechos]
    trechos = [t for t in trechos if len(t) >= 40]

    return paragrafos, trechos


def selecionar_candidatos(trechos, dados_referencia, threshold=SIMILARITY_THRESHOLD):
    """Etapa 1: filtra trechos por similaridade TF-IDF com as quotations humanas."""
    textos_referencia = dados_referencia["quotation"].tolist()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w{3,}\b",
    )
    vectorizer.fit(textos_referencia + trechos)

    tfidf_referencia = vectorizer.transform(textos_referencia)
    tfidf_novos = vectorizer.transform(trechos)

    similaridades = cosine_similarity(tfidf_novos, tfidf_referencia)

    max_similarity = similaridades.max(axis=1)
    indice_similar = similaridades.argmax(axis=1)
    quotation_similar = dados_referencia["quotation"].iloc[indice_similar].values

    candidatos = pd.DataFrame(
        {
            "trecho_id": range(1, len(trechos) + 1),
            "trecho": trechos,
            "similaridade": np.round(max_similarity, 4),
            "quotation_referencia": quotation_similar,
        }
    )

    candidatos = candidatos[candidatos["similaridade"] >= threshold].reset_index(
        drop=True
    )
    return candidatos


def classificar_trechos(
    textos,
    tokenizer,
    model,
    todas_tags,
    threshold=CLASSIFICATION_THRESHOLD,
    progress_callback=None,
):
    """Etapa 2: classifica os trechos candidatos com o BERT treinado.

    progress_callback, se fornecido, é chamado com um valor de 0 a 1
    a cada lote processado (útil para barra de progresso na interface).
    """
    resultados = []
    total_batches = max(1, (len(textos) + BATCH_SIZE - 1) // BATCH_SIZE)

    with torch.no_grad():
        for i, inicio in enumerate(range(0, len(textos), BATCH_SIZE)):
            fim = min(inicio + BATCH_SIZE, len(textos))
            batch = textos[inicio:fim]

            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            output = model(**enc)
            logits = output.logits.detach().cpu().numpy()

            probs = 1 / (1 + np.exp(-logits))
            resultados.append(probs)

            if progress_callback is not None:
                progress_callback((i + 1) / total_batches)

    probs_all = np.vstack(resultados)

    tags_preditas = []
    for x in probs_all:
        selecionadas = [
            todas_tags[i] for i in range(len(todas_tags)) if x[i] >= threshold
        ]
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


def analisar_texto_completo(
    texto_relatorio,
    tokenizer,
    model,
    todas_tags,
    dados_referencia,
    progress_callback=None,
):
    """Roda o pipeline completo (Etapa 1 + Etapa 2) sobre um texto de relatório.

    Retorna (resultados_finais, paragrafos, trechos, candidatos).
    """
    paragrafos, trechos = dividir_em_trechos(texto_relatorio)

    if len(trechos) == 0:
        return pd.DataFrame(), paragrafos, trechos, pd.DataFrame()

    candidatos = selecionar_candidatos(trechos, dados_referencia)

    if len(candidatos) == 0:
        return pd.DataFrame(), paragrafos, trechos, candidatos

    classificacoes = classificar_trechos(
        candidatos["trecho"].tolist(),
        tokenizer,
        model,
        todas_tags,
        progress_callback=progress_callback,
    )

    resultados_finais = pd.concat(
        [candidatos.reset_index(drop=True), classificacoes.drop(columns=["trecho"])],
        axis=1,
    )

    resultados_finais = resultados_finais.sort_values(
        "confianca", ascending=False
    ).reset_index(drop=True)

    return resultados_finais, paragrafos, trechos, candidatos
