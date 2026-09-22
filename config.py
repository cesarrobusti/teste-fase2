# ============================================================
# CONFIGURAÇÃO COMPARTILHADA
#
# Este arquivo é importado por todos os outros scripts do
# projeto, para que todos usem os mesmos caminhos e parâmetros.
#
# Ajuste os valores abaixo conforme seus arquivos.
# ============================================================

import os
from pathlib import Path

# ------------------------------------------------------------
# PASTA DE TRABALHO
#
# Detectada automaticamente como a pasta onde este arquivo
# (config.py) está salvo — NÃO é um caminho fixo. Funciona em
# qualquer computador e qualquer pasta, desde que todos os
# arquivos do projeto fiquem juntos na mesma pasta. Não precisa
# editar isto ao mover o projeto para outra máquina.
# ------------------------------------------------------------
DIR_TRABALHO = Path(__file__).resolve().parent
os.chdir(DIR_TRABALHO)

# ------------------------------------------------------------
# ARQUIVO COM AS QUOTATIONS JÁ CLASSIFICADAS
#
# Só é usado por treinar_modelo.py
# ------------------------------------------------------------
ARQUIVO_TREINO = "Artigo - Impact Tagging-quotations.xlsx"

# ------------------------------------------------------------
# QUOTATIONS REVISADAS (aprendizado ativo)
#
# Arquivo gerado automaticamente pelo app.py quando você revisa
# e corrige trechos na tela ("Revisar e corrigir"). Começa
# vazio/inexistente e vai crescendo com o uso.
#
# No próximo treino (treinar_modelo.py), estas correções são
# combinadas com ARQUIVO_TREINO automaticamente — não precisa
# copiar nada manualmente.
# ------------------------------------------------------------
ARQUIVO_REVISOES = "quotations_revisadas.xlsx"

# ------------------------------------------------------------
# NOVO RELATÓRIO
#
# Só é usado por analisar_relatorio.py.
# Basta trocar este arquivo quando quiser analisar outro.
# ------------------------------------------------------------
NOVO_RELATORIO = "Proposta_24009016_UNIVERSIDADE_FEDERAL_DE_CAMPINA_GRANDE_(UFCG).txt"

# ------------------------------------------------------------
# ARQUIVOS / DIRETÓRIOS
#
# Cada classificador salva em uma pasta separada, para você
# poder treinar e comparar os 3 sem que um sobrescreva o outro.
# ------------------------------------------------------------
DIR_MODELO = "./modelo_impacto"              # BERTimbau
DIR_MODELO_SETFIT = "./modelo_impacto_setfit"  # SetFit
DIR_MODELO_TFIDF = "./modelo_impacto_tfidf"    # TF-IDF + Regressão Logística
ARQUIVO_RESULTADO = "resultado_relatorio_159.xlsx"

# ------------------------------------------------------------
# CONFIGURAÇÕES DO SETFIT
# ------------------------------------------------------------
# Modelo base multilíngue (inclui português). Pequeno e rápido —
# bom ponto de partida. Alternativa maior/mais precisa (porém
# mais lenta): "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SETFIT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SETFIT_EPOCHS = 4
SETFIT_BATCH_SIZE = 16

# ------------------------------------------------------------
# MODELO EM PORTUGUÊS
#
# Só é usado por treinar_modelo.py (ponto de partida do
# treinamento). Depois de treinado, analisar_relatorio.py
# carrega o modelo já ajustado a partir de DIR_MODELO.
# ------------------------------------------------------------
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

# ------------------------------------------------------------
# CONFIGURAÇÕES DO BERT
# ------------------------------------------------------------
MAX_LENGTH = 256
# Com bases pequenas (poucas centenas de exemplos), o modelo
# precisa de mais passos de treino para sair do "chute
# uniforme". 4 épocas costuma ser pouco nesse cenário.
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
SEED = 42

# ------------------------------------------------------------
# TAGS RARAS
#
# Tags com poucos exemplos são estatisticamente impossíveis de
# aprender (e de avaliar) de forma confiável. Tags com menos
# de MIN_TAG_COUNT exemplos são excluídas do treino.
#
# Ideal: acumular mais quotations classificadas para essas
# tags, ou fundi-las com uma tag mais ampla no Excel de treino.
# ------------------------------------------------------------
MIN_TAG_COUNT = 5

# ------------------------------------------------------------
# PESO DAS CLASSES POSITIVAS NA FUNÇÃO DE PERDA
#
# Em problemas multi-label muito esparsos (poucas tags
# positivas por exemplo, entre muitas tags possíveis), o
# modelo tende a "chutar" probabilidade baixa para tudo,
# porque isso já minimiza bem o erro médio.
#
# Aplicamos pos_weight (peso maior para exemplos positivos,
# proporcional à raridade de cada tag) para forçar o modelo a
# se arriscar mais nas tags positivas. POS_WEIGHT_CAP limita o
# valor máximo do peso, para tags extremamente raras não
# desestabilizarem o treino.
# ------------------------------------------------------------
POS_WEIGHT_CAP = 15.0

# ------------------------------------------------------------
# ANÁLISE INTERPRETATIVA COM IA (Impacto AI)
#
# Cada pesquisador usa a PRÓPRIA chave de API do Google Gemini
# (gratuita, obtida em https://aistudio.google.com/). A chave
# nunca é salva em disco — fica só na sessão do navegador
# enquanto o app está aberto.
#
# GEMINI_MODEL_NAME: modelo gratuito atual. A Google descontinua
# modelos com frequência — se este parar de funcionar, veja o
# nome do modelo recomendado na mensagem de erro (costuma
# indicar o substituto) ou confira https://ai.google.dev/gemini-api/docs/models
# ------------------------------------------------------------
GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"

# Quantos trechos de maior confiança são enviados no prompt
# para a IA gerar o comentário interpretativo (mantém o prompt
# enxuto e barato).
GEMINI_MAX_TRECHOS_NO_PROMPT = 30

# ------------------------------------------------------------
# CONFIGURAÇÕES DA ETAPA 1 (localização de candidatos)
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
