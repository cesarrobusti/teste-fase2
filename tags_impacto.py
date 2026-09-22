# ============================================================
# TAGS DE IMPACTO — LISTA COMPLETA, ORGANIZADA POR BLOCO
#
# Esta é a lista de tags usada no seletor da tela de revisão do
# app (Passo ④ "Revisar, corrigir e ensinar o modelo").
#
# PARA ADICIONAR, REMOVER OU RENOMEAR UMA TAG:
# Edite o dicionário TAGS_IMPACTO abaixo. Cada chave é o nome de
# um bloco/categoria, e o valor é a lista de tags daquele bloco.
# Não precisa mexer em mais nenhum arquivo — o app lê esta lista
# automaticamente toda vez que é aberto.
#
# Formato final de cada tag no app: "Bloco: Nome da tag"
# (mesmo padrão usado no Excel de treino, ex.:
# "Áreas de Impacto: Econômica").
# ============================================================

TAGS_IMPACTO = {
    "Métodos de avaliação": [
        "Estudos de caso",
        "Entrevistas com stakeholders",
        "Pesquisas e questionários",
        "Análise de documentos",
        "Painéis de especialistas",
        "Análise bibliométrica",
        "Altmetria",
        "Métodos de avaliação econômica",
        "Avaliação participativa",
        "Escalas de pontuação e sistemas de classificação",
    ],
    "Categorias de Impacto": [
        "Impacto Instrumental",
        "Impacto Conceitual",
        "Impacto de Construção de Capacidade",
        "Impacto de Construção de Parcerias",
        "Impacto Intrauniversitário",
        "Impacto comunicacional",
        "Impacto Simbólico",
        "Impactos Relacionais/de Rede",
        "Impactos Culturais/Sistêmicos",
        "Efeitos Não Intencionais",
    ],
    "Áreas de Impacto": [
        "Produção Científica e de Conhecimento",
        "Econômica",
        "Ambiental",
        "Saúde e Bem-estar",
        "Política e Governança",
        "Social e Cultural",
        "Tecnologia e Inovação",
        "Organizacional",
        "Educacional",
    ],
    "Engajamento de Stakeholders": [
        "Identificação e Seleção de Stakeholders",
        "Interação Linear",
        "Interação Cíclica",
        "Coprodução",
        "Estratégias de Alinhamento",
        "Capacidade e Prontidão dos Stakeholders",
        "Priorização de Componentes de Avaliação",
        "Coleta de evidências",
        "Avaliação de impacto",
        "Validação e disseminação de resultados",
    ],
    "Nível de Influência Geográfica": [
        "Local",
        "Regional",
        "Nacional",
        "Internacional",
        "Todos os níveis",
    ],
}


def listar_tags_completas():
    """Retorna todas as tags no formato 'Bloco: Tag', agrupadas por
    bloco na ordem em que aparecem em TAGS_IMPACTO (para o seletor
    do app mostrar as opções organizadas por categoria)."""
    tags = []
    for bloco, lista_tags in TAGS_IMPACTO.items():
        for tag in lista_tags:
            tags.append(f"{bloco}: {tag}")
    return tags
