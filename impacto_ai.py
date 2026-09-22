# ============================================================
# IMPACTO AI — ANÁLISE INTERPRETATIVA COM IA (Google Gemini)
#
# Assistente de IA integrado a este programa: gera um comentário
# interpretativo sobre os resultados da classificação. Cada
# pesquisador usa a PRÓPRIA chave de API do Google Gemini,
# obtida gratuitamente em:
#
#     https://aistudio.google.com/
#
# A chave nunca é salva em disco por este programa — ela vive
# apenas na sessão do navegador (st.session_state) enquanto o
# app está aberto, e é descartada ao fechar a aba.
#
# IMPORTANTE: usa o SDK novo e unificado do Google, o pacote
# "google-genai" (import "from google import genai"). O pacote
# antigo "google-generativeai" foi descontinuado pelo Google —
# não use `import google.generativeai`, que é a versão legada.
#
# Não precisa rodar este arquivo diretamente.
# ============================================================

from google import genai
import pandas as pd

from config import GEMINI_MODEL_NAME, GEMINI_MAX_TRECHOS_NO_PROMPT


class ChaveGeminiInvalida(Exception):
    """Erro amigável para quando a chave de API não funciona."""


def validar_chave(api_key):
    """Faz uma chamada mínima para confirmar que a chave funciona.

    Levanta ChaveGeminiInvalida com uma mensagem amigável em caso
    de erro (chave errada, cota esgotada, sem internet, modelo
    descontinuado, etc.).
    """
    if not api_key or not api_key.strip():
        raise ChaveGeminiInvalida("Nenhuma chave foi informada.")

    try:
        client = genai.Client(api_key=api_key.strip())
        client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents="Responda apenas: ok",
        )
    except Exception as e:
        raise ChaveGeminiInvalida(_mensagem_amigavel(e))


def _mensagem_amigavel(erro):
    texto = str(erro).lower()

    if "api_key" in texto or "api key" in texto or "invalid" in texto or "unauthenticated" in texto:
        return (
            "Chave inválida. Confira se você copiou a chave completa de "
            "aistudio.google.com."
        )
    if "quota" in texto or "429" in texto or "resource_exhausted" in texto:
        return (
            "Limite de uso gratuito atingido para esta chave. Tente novamente "
            "mais tarde, ou gere uma nova chave em aistudio.google.com."
        )
    if "permission" in texto or "403" in texto:
        return "Chave sem permissão para este modelo. Verifique em aistudio.google.com."
    if "404" in texto or "not found" in texto or "no longer available" in texto:
        return (
            f"O modelo '{GEMINI_MODEL_NAME}' não está mais disponível segundo o "
            "Google. Atualize GEMINI_MODEL_NAME em config.py para o modelo "
            "indicado na mensagem de erro do Google, ou confira a lista atual "
            "em https://ai.google.dev/gemini-api/docs/models"
        )

    return f"Não foi possível validar a chave: {erro}"


def _montar_prompt(resultados_finais, frequencia_tags, nome_relatorio):
    """Monta um prompt detalhado com o resumo estruturado da análise —
    não o relatório inteiro — para manter o custo e a chance de
    alucinação baixos.

    ==========================================================
    ONDE EDITAR O PROMPT (para ajustar o comentário gerado):
    A string `prompt`, logo abaixo, é o texto enviado à IA. Edite
    livremente as instruções finais (a partir de "Com base nas
    informações acima...") para pedir outro formato, tom, nível
    de detalhe, ou focar em outros aspectos.
    ==========================================================
    """
    # Só os trechos com tag prevista, ordenados por confiança,
    # limitados a GEMINI_MAX_TRECHOS_NO_PROMPT.
    trechos_com_tag = resultados_finais[resultados_finais["tags_preditas"].notna()]
    trechos_com_tag = trechos_com_tag.sort_values("confianca", ascending=False)
    trechos_com_tag = trechos_com_tag.head(GEMINI_MAX_TRECHOS_NO_PROMPT)

    linhas_trechos = []
    for _, linha in trechos_com_tag.iterrows():
        linhas_trechos.append(
            f"- Trecho: \"{linha['trecho'][:400]}\"\n"
            f"  Tags: {linha['tags_preditas']} (confiança: {linha['confianca']:.0%})"
        )
    bloco_trechos = "\n".join(linhas_trechos) if linhas_trechos else "(nenhum trecho classificado)"

    # --------------------------------------------------------
    # Frequência das tags NESTE relatório (não na base de treino
    # geral) — é isso que torna o comentário específico ao
    # relatório analisado, em vez de genérico.
    # --------------------------------------------------------
    tags_do_relatorio = []
    for tags_texto in resultados_finais["tags_preditas"].dropna():
        tags_do_relatorio.extend(t.strip() for t in str(tags_texto).split("|") if t.strip())

    if tags_do_relatorio:
        contagem = pd.Series(tags_do_relatorio).value_counts()
        bloco_tags_relatorio = "\n".join(f"- {tag}: {n} trecho(s)" for tag, n in contagem.items())
    else:
        bloco_tags_relatorio = "(nenhuma tag identificada neste relatório)"

    # Tags que o modelo conhece mas que NÃO apareceram neste
    # relatório — usadas para apontar lacunas.
    todas_tags_conhecidas = set(frequencia_tags["tag"].tolist())
    tags_ausentes = sorted(todas_tags_conhecidas - set(tags_do_relatorio))
    bloco_tags_ausentes = "\n".join(f"- {tag}" for tag in tags_ausentes) if tags_ausentes else "(nenhuma — todas as tags apareceram)"

    n_total = len(resultados_finais)
    n_revisar = int(resultados_finais["revisar"].sum()) if len(resultados_finais) else 0

    prompt = f"""Você é um assistente de pesquisa especializado em avaliação de impacto \
científico, ajudando a interpretar os resultados de um classificador automático \
(modelo BERTimbau treinado por um pesquisador) que identifica trechos de relatórios \
com evidências de impacto de pesquisa e atribui tags a cada trecho.

Relatório analisado: {nome_relatorio}
Total de trechos classificados: {n_total}
Trechos marcados para revisão humana (baixa confiança): {n_revisar}

Distribuição das tags identificadas NESTE relatório (quantos trechos de cada tipo):
{bloco_tags_relatorio}

Tags que o modelo conhece mas que NÃO apareceram neste relatório (possíveis lacunas):
{bloco_tags_ausentes}

Trechos deste relatório com maior confiança de classificação (evidência textual):
{bloco_trechos}

Com base nas informações acima, escreva um comentário interpretativo completo e \
detalhado (organize em seções com subtítulos, em português), cobrindo:

1. **Visão geral**: um resumo dos tipos de impacto predominantes neste relatório \
específico, citando os números da distribuição acima (não fale em termos vagos — \
use as quantidades reais).

2. **Impactos em destaque**: para as 2-3 tags mais frequentes neste relatório, \
comente o que os trechos de maior confiança sugerem sobre COMO esse impacto está \
sendo gerado nesta pesquisa específica (baseando-se no conteúdo dos trechos citados).

3. **Lacunas identificadas**: liste os tipos de impacto pouco ou nada representados \
neste relatório (use a lista de tags ausentes acima). Para CADA lacuna relevante, \
sugira 2-3 exemplos PRÁTICOS e CONCRETOS de atividades, ações ou evidências que o \
pesquisador poderia buscar ou documentar para gerar esse tipo de impacto — baseado \
no seu conhecimento geral sobre avaliação de impacto de pesquisa (não invente fatos \
sobre este relatório específico; deixe claro quando a sugestão é uma recomendação \
geral, não uma constatação sobre o relatório).

4. **Observação final**: mencione, de forma clara, que esta é uma interpretação \
automática que não substitui a revisão humana, especialmente dos trechos marcados \
para revisão.

Seja específico e evite frases genéricas que serviriam para qualquer relatório — \
ancore cada afirmação nos números e trechos fornecidos acima."""

    return prompt


def gerar_comentario(api_key, resultados_finais, frequencia_tags, nome_relatorio):
    """Gera o comentário interpretativo com o Gemini.

    Retorna o texto gerado (string). Levanta ChaveGeminiInvalida em
    caso de erro de autenticação/cota/rede/modelo indisponível.
    """
    try:
        client = genai.Client(api_key=api_key.strip())

        prompt = _montar_prompt(resultados_finais, frequencia_tags, nome_relatorio)
        resposta = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )

        return resposta.text
    except ChaveGeminiInvalida:
        raise
    except Exception as e:
        raise ChaveGeminiInvalida(_mensagem_amigavel(e))
