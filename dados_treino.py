# ============================================================
# PREPARAÇÃO DE DADOS — COMPARTILHADA ENTRE OS 3 CLASSIFICADORES
#
# Este módulo é importado por:
#   - treinar_modelo.py          (BERTimbau)
#   - treinar_modelo_setfit.py   (SetFit)
#   - treinar_modelo_tfidf.py    (TF-IDF + Regressão Logística)
#
# Garante que os três modelos treinem e sejam avaliados no
# MESMO conjunto de treino/teste — condição necessária para a
# comparação entre eles ser justa.
#
# Não precisa rodar este arquivo diretamente.
# ============================================================

import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import ARQUIVO_TREINO, ARQUIVO_REVISOES, SEED, MIN_TAG_COUNT

# Ao mover exemplos para o teste, garante que sobre pelo menos
# este número de exemplos de cada tag no treino.
MIN_TRAIN_POR_TAG = 2


def separar_tags(x):
    x = str(x)
    tags = re.split(r"\s*[,;|]\s*", x)
    tags = [t.strip() for t in tags if t and t.strip()]
    vistos = set()
    resultado = []
    for t in tags:
        if t not in vistos:
            vistos.add(t)
            resultado.append(t)
    return resultado


def split_estratificado_multilabel(
    label_matrix, test_size=0.20, seed=SEED, min_train_por_tag=MIN_TRAIN_POR_TAG
):
    """Divide em treino/teste tentando manter pelo menos
    min_train_por_tag exemplos de cada tag no treino, mesmo com
    tags raras — evita que uma tag rara "suma" do treino por
    azar do sorteio aleatório.
    """
    rng = np.random.default_rng(seed)
    n = label_matrix.shape[0]
    ordem = rng.permutation(n)

    alvo_teste = round(n * test_size)
    contagem_treino = label_matrix.sum(axis=0).astype(int)

    test_idx = []
    for i in ordem:
        if len(test_idx) >= alvo_teste:
            break

        tags_do_exemplo = np.where(label_matrix[i] == 1)[0]

        if len(tags_do_exemplo) == 0:
            pode_mover = True
        else:
            pode_mover = all(
                contagem_treino[t] - 1 >= min_train_por_tag for t in tags_do_exemplo
            )

        if pode_mover:
            test_idx.append(i)
            for t in tags_do_exemplo:
                contagem_treino[t] -= 1

    test_idx = np.array(sorted(test_idx))
    train_idx = np.array(sorted(set(range(n)) - set(test_idx)))

    return train_idx, test_idx


def carregar_dados_treino(verbose=True):
    """Lê, limpa e prepara os dados de treino.

    Retorna um dicionário com:
        dados               DataFrame limpo, com coluna 'tags'
        todas_tags          lista de tags (após filtro de raridade)
        frequencia_tags     DataFrame tag/n (após filtro)
        tags_removidas      DataFrame tag/n das tags removidas por raridade
        label_matrix        matriz multi-label (todas as linhas)
        train_idx, test_idx índices do split
        train_texts, test_texts
        train_labels, test_labels
    """

    def log(*args):
        if verbose:
            print(*args)

    # --------------------------------------------------------
    # 1. LER AS QUOTATIONS CLASSIFICADAS
    #
    # Combina o Excel de treino original com as correções
    # salvas pelo app.py (aprendizado ativo), se existirem.
    # Quando o mesmo trecho aparece nos dois arquivos, a versão
    # revisada por um humano prevalece sobre a original.
    # --------------------------------------------------------
    dados = pd.read_excel(ARQUIVO_TREINO)

    if "quotation" not in dados.columns:
        raise ValueError("A coluna 'quotation' não foi encontrada.")
    if "codes" not in dados.columns:
        raise ValueError("A coluna 'codes' não foi encontrada.")

    dados = dados[["quotation", "codes"]].copy()

    if Path(ARQUIVO_REVISOES).exists():
        revisoes = pd.read_excel(ARQUIVO_REVISOES)

        if "quotation" in revisoes.columns and "codes" in revisoes.columns:
            revisoes = revisoes[["quotation", "codes"]].copy()

            n_antes = len(dados)
            # concat com as revisões por último: no drop_duplicates abaixo,
            # keep="last" faz a versão revisada vencer quando o mesmo
            # trecho aparece nos dois arquivos.
            dados = pd.concat([dados, revisoes], ignore_index=True)
            dados["quotation"] = dados["quotation"].astype(str)
            dados = dados.drop_duplicates(subset="quotation", keep="last")

            log(
                f"Quotations revisadas incorporadas: {len(revisoes)} "
                f"(base cresceu de {n_antes} para {len(dados)} quotations únicas)"
            )
        else:
            log(
                f"⚠️  '{ARQUIVO_REVISOES}' encontrado, mas sem as colunas "
                "'quotation'/'codes' esperadas — ignorando."
            )

    # --------------------------------------------------------
    # 2. LIMPEZA
    # --------------------------------------------------------
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

    log("Quotations classificadas:", len(dados))

    # --------------------------------------------------------
    # 3. TRANSFORMAR CODES EM TAGS
    # --------------------------------------------------------
    dados["tags"] = dados["codes"].apply(separar_tags)

    # --------------------------------------------------------
    # 4/5. LISTA DE TAGS E FREQUÊNCIA (antes do filtro)
    # --------------------------------------------------------
    todas_tags_bruta = sorted(set(tag for tags in dados["tags"] for tag in tags))

    frequencia_tags_bruta = pd.DataFrame(
        {
            "tag": todas_tags_bruta,
            "n": [
                sum(tag in tags for tags in dados["tags"]) for tag in todas_tags_bruta
            ],
        }
    ).sort_values("n", ascending=False).reset_index(drop=True)

    # --------------------------------------------------------
    # 5b. FILTRAR TAGS RARAS
    # --------------------------------------------------------
    tags_removidas = frequencia_tags_bruta[frequencia_tags_bruta["n"] < MIN_TAG_COUNT]

    if len(tags_removidas) > 0 and verbose:
        log(
            f"\n⚠️  {len(tags_removidas)} tag(s) removida(s) por ter(em) menos de "
            f"{MIN_TAG_COUNT} exemplos:"
        )
        log(tags_removidas)

    todas_tags = sorted(
        frequencia_tags_bruta[frequencia_tags_bruta["n"] >= MIN_TAG_COUNT][
            "tag"
        ].tolist()
    )

    dados["tags"] = dados["tags"].apply(lambda tags: [t for t in tags if t in todas_tags])

    frequencia_tags = frequencia_tags_bruta[
        frequencia_tags_bruta["tag"].isin(todas_tags)
    ].reset_index(drop=True)

    log("\nNúmero de tags (após filtro):", len(todas_tags))

    # --------------------------------------------------------
    # 6. MATRIZ MULTI-LABEL
    # --------------------------------------------------------
    label_matrix = np.zeros((len(dados), len(todas_tags)), dtype=np.int64)
    tag_to_idx = {tag: i for i, tag in enumerate(todas_tags)}

    for i, tags in enumerate(dados["tags"]):
        for tag in tags:
            if tag in tag_to_idx:
                label_matrix[i, tag_to_idx[tag]] = 1

    # --------------------------------------------------------
    # 7. DIVISÃO TRAIN / TEST (estratificada por tag)
    # --------------------------------------------------------
    train_idx, test_idx = split_estratificado_multilabel(label_matrix)

    train_texts = dados["quotation"].iloc[train_idx].tolist()
    test_texts = dados["quotation"].iloc[test_idx].tolist()
    train_labels = label_matrix[train_idx]
    test_labels = label_matrix[test_idx]

    log("\nTreinamento:", len(train_texts))
    log("Teste:", len(test_texts))

    tags_sem_exemplo_treino = [
        todas_tags[i] for i in range(len(todas_tags)) if train_labels[:, i].sum() == 0
    ]
    if tags_sem_exemplo_treino and verbose:
        log(
            "\n⚠️  As tags abaixo ficaram sem nenhum exemplo no treino (raras "
            "demais). Nenhum dos 3 modelos vai aprender a prever essas tags:"
        )
        log(tags_sem_exemplo_treino)

    return {
        "dados": dados,
        "todas_tags": todas_tags,
        "frequencia_tags": frequencia_tags,
        "tags_removidas": tags_removidas,
        "label_matrix": label_matrix,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_texts": train_texts,
        "test_texts": test_texts,
        "train_labels": train_labels,
        "test_labels": test_labels,
    }


def salvar_revisoes(pares_trecho_tags):
    """Adiciona novas quotations revisadas/corrigidas ao arquivo de
    revisões (ARQUIVO_REVISOES), para entrarem no próximo treino.

    pares_trecho_tags: lista de tuplas (trecho, tags), onde tags é uma
    lista de strings (ex.: ["Áreas de impacto: Econômica", "..."]).

    Se o mesmo trecho já existir no arquivo de revisões, a versão mais
    recente (a que está sendo salva agora) substitui a anterior.

    Retorna o número de linhas efetivamente salvas.
    """
    linhas_novas = [
        {"quotation": trecho, "codes": " | ".join(tags)}
        for trecho, tags in pares_trecho_tags
        if trecho and tags
    ]

    if not linhas_novas:
        return 0

    novo_df = pd.DataFrame(linhas_novas)

    caminho = Path(ARQUIVO_REVISOES)
    if caminho.exists():
        existentes = pd.read_excel(caminho)
        combinado = pd.concat([existentes, novo_df], ignore_index=True)
    else:
        combinado = novo_df

    combinado["quotation"] = combinado["quotation"].astype(str)
    combinado = combinado.drop_duplicates(subset="quotation", keep="last")

    combinado.to_excel(caminho, index=False)

    return len(linhas_novas)
