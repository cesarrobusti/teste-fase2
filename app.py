# ============================================================
# APP VISUAL — CLASSIFICADOR DE IMPACTO DE PESQUISA
#
# Interface para pesquisadores enviarem um relatório (.txt ou
# .docx) e obterem os tipos de impacto identificados pelo
# modelo já treinado (BERTimbau).
#
# NÃO EXIGE CONHECIMENTO DE PROGRAMAÇÃO PARA USAR — basta
# seguir os passos numerados na tela.
#
# COMO INSTALAR E RODAR (uma vez só, quem for configurar):
#
#   1) Coloque estes arquivos na mesma pasta:
#        - app.py
#        - impacto_core.py
#        - config.py
#        - requirements.txt
#        - a pasta "modelo_impacto" (modelo já treinado)
#        - o Excel de quotations de referência (ARQUIVO_TREINO
#          definido em config.py)
#
#   2) Crie e ative um ambiente virtual, depois instale as
#      dependências:
#
#         python -m venv .venv
#         .venv\\Scripts\\activate
#         pip install -r requirements.txt
#
#   3) Depois disso, para ABRIR O APP no dia a dia, dá para
#      usar o atalho "iniciar_app.bat" (basta dar duplo clique)
#      em vez de digitar comandos — veja esse arquivo.
#
#      Ou, manualmente, no terminal:
#
#         streamlit run app.py
#
#   4) O navegador abre automaticamente em algo como
#      http://localhost:8501
# ============================================================

import io
from datetime import datetime

import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection
from config import DIR_MODELO, SIMILARITY_THRESHOLD, CLASSIFICATION_THRESHOLD
import impacto_core as core
from dados_treino import salvar_revisoes
import impacto_ai
import tags_impacto

st.set_page_config(
    page_title="Classificador de Impacto de Pesquisa",
    page_icon="📊",
    layout="wide",
)

# ------------------------------------------------------------
# CARREGAMENTO DO MODELO E DA REFERÊNCIA (uma vez, com cache)
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo treinado...")
def get_modelo():
    tokenizer, model = core.carregar_modelo()
    tags = core.carregar_tags()
    return tokenizer, model, tags


@st.cache_data(show_spinner="Carregando quotations de referência...")
def get_referencia():
    return core.carregar_referencia()


# ------------------------------------------------------------
# CABEÇALHO
# ------------------------------------------------------------
st.title("📊 Classificador de Impacto de Pesquisa")
st.caption(
    "Envie um relatório para identificar automaticamente os trechos com "
    "evidências de impacto e suas respectivas tags. Não é necessário "
    "nenhum conhecimento técnico — basta seguir os passos abaixo."
)

if not core.modelo_disponivel():
    st.error(
        f"⚠️ Não encontrei um modelo treinado na pasta `{DIR_MODELO}`.\n\n"
        "Peça para quem configurou o sistema verificar se a pasta "
        f"`{DIR_MODELO}` (com o modelo já treinado) está no lugar certo, "
        "junto com os demais arquivos do programa."
    )
    st.stop()

tokenizer, model, todas_tags = get_modelo()
dados_referencia = get_referencia()

with st.sidebar:
    st.header("ℹ️ Sobre")
    st.write(f"**Tags que o modelo reconhece:** {len(todas_tags)}")
    with st.expander("Ver lista de tags"):
        for tag in todas_tags:
            st.write("•", tag)

    st.divider()
    st.header("🧠 Impacto AI (análise interpretativa)")
    st.caption(
        "Opcional. Use sua própria chave gratuita da API do Google Gemini "
        "para gerar um comentário interpretativo automático sobre os "
        "resultados da classificação."
    )

    with st.expander("❓ Como obter e usar sua chave gratuita"):
        st.markdown(
            """
**1.** Acesse [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
e faça login com uma conta Google (a mesma do Gmail já serve).

**2.** Clique em **"Create API key"** (ou "Criar chave de API"). Não é
necessário cartão de crédito nem plano pago — a camada gratuita é
suficiente para uso individual.

**3.** Copie a chave gerada (um texto começando com `AIza...`).

**4.** Cole a chave no campo **"Sua chave de API do Gemini"** logo abaixo
e clique em **"Validar chave"** para confirmar que está funcionando.

**5.** Pronto — o botão **"🧠 Gerar comentário interpretativo"** (passo
⑤, na tela principal, depois de analisar um relatório) fica liberado.

**Importante:**
- A chave é **sua e gratuita** — cada pesquisador usa a própria, sem
  custo para quem distribuiu este programa.
- Ela **não é salva** em nenhum arquivo por este programa — vale só
  enquanto esta aba do navegador estiver aberta. Você pode reutilizar a
  mesma chave sempre que quiser, sem precisar gerar uma nova a cada uso.
- Se aparecer erro de "limite atingido", é porque a cota diária
  gratuita da sua chave acabou — tente novamente mais tarde.
            """
        )

    st.markdown(
        "[🔑 Obter chave gratuita em aistudio.google.com](https://aistudio.google.com/apikey)"
    )

    gemini_api_key_input = st.text_input(
        "Sua chave de API do Gemini",
        type="password",
        value=st.session_state.get("gemini_api_key", ""),
        placeholder="Cole sua chave aqui (AIza...)",
        help=(
            "A chave fica só nesta sessão do navegador — nunca é salva em "
            "disco por este programa. Feche a aba para descartá-la."
        ),
    )

    if gemini_api_key_input != st.session_state.get("gemini_api_key", ""):
        st.session_state["gemini_api_key"] = gemini_api_key_input
        st.session_state["gemini_key_validada"] = False

    if st.session_state.get("gemini_api_key"):
        if st.button("Validar chave", use_container_width=True):
            with st.spinner("Validando chave..."):
                try:
                    impacto_ai.validar_chave(st.session_state["gemini_api_key"])
                    st.session_state["gemini_key_validada"] = True
                    st.success("✅ Chave válida.")
                except impacto_ai.ChaveGeminiInvalida as e:
                    st.session_state["gemini_key_validada"] = False
                    st.error(f"❌ {e}")

    st.divider()
    st.header("⚙️ Parâmetros avançados")
    st.caption(
        "Só mexa aqui se souber o que está fazendo. Os valores padrão "
        "funcionam bem na maioria dos casos."
    )
    similarity_threshold = st.slider(
        "Similaridade mínima (Etapa 1)",
        min_value=0.05,
        max_value=0.50,
        value=float(SIMILARITY_THRESHOLD),
        step=0.01,
        help=(
            "Trechos com similaridade abaixo deste valor, em relação às "
            "quotations humanas, não são considerados candidatos."
        ),
    )
    classification_threshold = st.slider(
        "Threshold de classificação (Etapa 2)",
        min_value=0.10,
        max_value=0.90,
        value=float(CLASSIFICATION_THRESHOLD),
        step=0.05,
        help="Probabilidade mínima para uma tag ser atribuída a um trecho.",
    )

# ------------------------------------------------------------
# PASSO 1 — ENTRADA DO(S) RELATÓRIO(S)
# ------------------------------------------------------------
st.subheader("① Envie o(s) relatório(s)")

modo = st.radio(
    "Como você quer enviar o texto?",
    ["📎 Enviar arquivo(s) (.txt ou .docx)", "📋 Colar texto diretamente"],
    horizontal=True,
    label_visibility="collapsed",
)

documentos = []  # lista de {"nome": ..., "texto": ...}

if modo.startswith("📎"):
    arquivos = st.file_uploader(
        "Selecione um ou mais arquivos de relatório",
        type=["txt", "docx", "doc"],
        accept_multiple_files=True,
        help="Formatos aceitos: .txt e .docx. Arquivos .doc antigos precisam ser convertidos para .docx no Word primeiro. Você pode selecionar vários arquivos de uma vez.",
    )

    if arquivos:
        for arquivo in arquivos:
            try:
                texto_extraido = core.extrair_texto(arquivo.name, arquivo.read())
                nome_doc = arquivo.name.rsplit(".", 1)[0]

                if not texto_extraido.strip():
                    st.warning(f"⚠️ '{arquivo.name}' parece estar vazio — ignorado.")
                    continue

                documentos.append({"nome": nome_doc, "texto": texto_extraido})
            except ValueError as e:
                st.error(f"❌ '{arquivo.name}': {e}")

        if documentos:
            st.success(
                f"✅ {len(documentos)} arquivo(s) carregado(s): "
                + ", ".join(d["nome"] for d in documentos)
            )
else:
    texto_colado = st.text_area(
        "Cole aqui o texto do relatório",
        height=250,
        placeholder="Cole o texto do relatório aqui...",
    )
    if texto_colado.strip():
        documentos.append({"nome": "texto_colado", "texto": texto_colado})

# ------------------------------------------------------------
# PASSO 2 — ANÁLISE
# ------------------------------------------------------------
st.subheader("② Analisar")

analisar = st.button(
    "🔍 Analisar relatório(s)",
    type="primary",
    disabled=(len(documentos) == 0),
    use_container_width=True,
)

if len(documentos) == 0:
    st.caption("Envie um ou mais arquivos, ou cole um texto no passo ① para habilitar este botão.")

if analisar and documentos:
    progress_bar = st.progress(0.0, text="Iniciando análise...")

    resultados_por_documento = []
    total_documentos = len(documentos)

    for i_doc, doc in enumerate(documentos):
        nome_doc = doc["nome"]
        texto_relatorio = doc["texto"]

        def atualizar_progresso(fracao, nome_doc=nome_doc, i_doc=i_doc):
            fracao_global = (i_doc + fracao) / total_documentos
            progress_bar.progress(
                fracao_global,
                text=f"[{nome_doc}] Classificando trechos... {int(fracao * 100)}%",
            )

        progress_bar.progress(
            i_doc / total_documentos, text=f"[{nome_doc}] Localizando trechos candidatos..."
        )

        paragrafos, trechos = core.dividir_em_trechos(texto_relatorio)

        if len(trechos) == 0:
            st.warning(f"⚠️ '{nome_doc}': não foi possível identificar trechos analisáveis.")
            continue

        candidatos = core.selecionar_candidatos(
            trechos, dados_referencia, threshold=similarity_threshold
        )

        if len(candidatos) == 0:
            st.warning(
                f"⚠️ '{nome_doc}': nenhum trecho ultrapassou o threshold de "
                f"similaridade ({similarity_threshold})."
            )
            continue

        classificacoes = core.classificar_trechos(
            candidatos["trecho"].tolist(),
            tokenizer,
            model,
            todas_tags,
            threshold=classification_threshold,
            progress_callback=atualizar_progresso,
        )

        resultado_doc = pd.concat(
            [
                candidatos.reset_index(drop=True),
                classificacoes.drop(columns=["trecho"]),
            ],
            axis=1,
        )
        resultado_doc.insert(0, "arquivo", nome_doc)

        resultados_por_documento.append(resultado_doc)

    progress_bar.empty()

    if not resultados_por_documento:
        st.warning("Nenhum trecho classificado em nenhum dos documentos enviados.")
    else:
        resultados_finais = pd.concat(resultados_por_documento, ignore_index=True)
        resultados_finais = resultados_finais.sort_values(
            "confianca", ascending=False
        ).reset_index(drop=True)

        st.session_state["resultados_finais"] = resultados_finais
        st.session_state["nome_relatorio"] = (
            documentos[0]["nome"]
            if len(documentos) == 1
            else f"{len(documentos)}_relatorios"
        )
        st.session_state["nomes_documentos"] = [d["nome"] for d in documentos]

        n_docs_com_resultado = resultados_finais["arquivo"].nunique()
        st.success(
            f"✅ Análise concluída: {n_docs_com_resultado} de {total_documentos} "
            f"arquivo(s) com trechos classificados, "
            f"{len(resultados_finais)} trechos no total."
        )

# ------------------------------------------------------------
# PASSO 3 — RESULTADOS
# ------------------------------------------------------------
if "resultados_finais" in st.session_state:
    st.subheader("③ Resultados")

    resultados_finais = st.session_state["resultados_finais"]
    nomes_documentos = st.session_state.get("nomes_documentos", [])

    col1, col2, col3 = st.columns(3)
    col1.metric("Trechos classificados", len(resultados_finais))
    col2.metric(
        "Para revisão humana",
        int(resultados_finais["revisar"].sum()) if len(resultados_finais) else 0,
    )
    col3.metric(
        "Confiança média",
        f"{resultados_finais['confianca'].mean():.0%}" if len(resultados_finais) else "—",
    )

    if len(nomes_documentos) > 1:
        arquivos_disponiveis = ["Todos"] + sorted(resultados_finais["arquivo"].unique().tolist())
        filtro_arquivo = st.selectbox("Filtrar por arquivo", arquivos_disponiveis)
    else:
        filtro_arquivo = "Todos"

    apenas_com_tag = st.checkbox(
        "Mostrar apenas trechos com alguma tag prevista", value=True
    )

    tabela_exibicao = resultados_finais[
        ["arquivo", "trecho", "similaridade", "tags_preditas", "confianca", "numero_tags", "revisar"]
    ]
    if filtro_arquivo != "Todos":
        tabela_exibicao = tabela_exibicao[tabela_exibicao["arquivo"] == filtro_arquivo]
    if apenas_com_tag:
        tabela_exibicao = tabela_exibicao[tabela_exibicao["tags_preditas"].notna()]

    st.dataframe(tabela_exibicao, use_container_width=True, height=450)

    # ----------------------------------------------------------
    # DOWNLOAD DOS RESULTADOS — EXCEL E CSV
    # ----------------------------------------------------------
    st.markdown("**⬇️ Baixar resultados**")

    frequencia_tags = core.carregar_frequencia_tags()
    carimbo = datetime.now().strftime("%Y%m%d_%H%M")
    nome_base = st.session_state.get("nome_relatorio", "relatorio")

    col_xlsx, col_csv = st.columns(2)

    with col_xlsx:
        buffer_xlsx = io.BytesIO()
        with pd.ExcelWriter(buffer_xlsx, engine="openpyxl") as writer:
            resultados_finais.to_excel(writer, sheet_name="resultados", index=False)
            frequencia_tags.to_excel(writer, sheet_name="frequencia_tags", index=False)
        buffer_xlsx.seek(0)

        st.download_button(
            "📊 Baixar Excel (.xlsx) — recomendado",
            data=buffer_xlsx,
            file_name=f"resultado_{nome_base}_{carimbo}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.caption("Contém duas abas: resultados e frequência das tags.")

    with col_csv:
        buffer_csv = resultados_finais.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "📄 Baixar CSV (.csv)",
            data=buffer_csv,
            file_name=f"resultado_{nome_base}_{carimbo}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Apenas a tabela de resultados (sem a aba de frequência).")

    # ----------------------------------------------------------
  # ----------------------------------------------------------
    # PASSO ④ — REVISAR, CORRIGIR E ENSINAR O MODELO
    #
    # Aprendizado ativo: você corrige os trechos que o modelo
    # classificou errado ou com baixa confiança, e essas
    # correções ficam salvas para entrar no próximo treino
    # no Google Sheets.
    # ----------------------------------------------------------
    st.divider()
    st.subheader("④ Revisar, corrigir e ensinar o modelo (opcional)")
    st.caption(
        "Corrija as tags dos trechos abaixo quando o modelo errar ou tiver "
        "pouca confiança. As correções ficam guardadas e entram automaticamente "
        "no próximo treino do modelo — ele aprende com seus ajustes."
    )

    apenas_revisao = st.checkbox(
        "Mostrar apenas trechos marcados para revisão (confiança baixa)",
        value=True,
        key="apenas_revisao_checkbox",
    )

    base_revisao = resultados_finais.copy()
    if apenas_revisao:
        base_revisao = base_revisao[base_revisao["revisar"]]

    if len(base_revisao) == 0:
        st.info("Nenhum trecho para revisar com o filtro atual.")
    else:
        tags_disponiveis = tags_impacto.listar_tags_completas()

        st.caption(
            "Coluna **Tag prevista pelo modelo**: o que o modelo identificou. "
            "Coluna **Tags corrigidas**: clique para selecionar a(s) tag(s) certas "
            "na lista (digite para buscar), ou digite uma tag nova e pressione Enter "
            "se ela não estiver na lista. Marque **Incluir no próximo treino** "
            "apenas nas linhas que você já revisou."
        )

        def _parse_tags(texto):
            if pd.isna(texto):
                return []
            return [t.strip() for t in str(texto).split("|") if t.strip()]

        tabela_revisao = base_revisao[["arquivo", "trecho", "tags_preditas", "confianca"]].copy()
        tabela_revisao = tabela_revisao.rename(
            columns={"tags_preditas": "tag_prevista_pelo_modelo"}
        )
        tabela_revisao["tags_corrigidas"] = tabela_revisao["tag_prevista_pelo_modelo"].apply(
            lambda texto: [t for t in _parse_tags(texto) if t in tags_disponiveis]
        )
        tabela_revisao["incluir_no_treino"] = False

        tabela_editada = st.data_editor(
            tabela_revisao,
            use_container_width=True,
            height=450,
            num_rows="fixed",
            disabled=["arquivo", "trecho", "tag_prevista_pelo_modelo", "confianca"],
            column_order=[
                "arquivo",
                "trecho",
                "tag_prevista_pelo_modelo",
                "tags_corrigidas",
                "confianca",
                "incluir_no_treino",
            ],
            column_config={
                "arquivo": st.column_config.TextColumn("Arquivo", width="small"),
                "trecho": st.column_config.TextColumn("Trecho", width="large"),
                "tag_prevista_pelo_modelo": st.column_config.TextColumn(
                    "Tag prevista pelo modelo", width="medium"
                ),
                "tags_corrigidas": st.column_config.MultiselectColumn(
                    "Tags corrigidas",
                    options=tags_disponiveis,
                    accept_new_options=True,
                    width="large",
                ),
                "confianca": st.column_config.NumberColumn(
                    "Confiança", format="%.2f"
                ),
                "incluir_no_treino": st.column_config.CheckboxColumn(
                    "Incluir no próximo treino"
                ),
            },
            key="editor_revisao",
        )

        if st.button("💾 Salvar correções para o próximo treino", type="primary"):
            selecionadas = tabela_editada[tabela_editada["incluir_no_treino"]]

            linhas_para_salvar = []
            agora = datetime.now().strftime("%Y-%m-%d %H:%M")

            for _, linha in selecionadas.iterrows():
                tags = linha["tags_corrigidas"]
                if isinstance(tags, list) and tags:
                    tags_formatadas = " | ".join(tags)
                    linhas_para_salvar.append({
                        "arquivo": linha["arquivo"],
                        "trecho": linha["trecho"],
                        "tags_corrigidas": tags_formatadas,
                        "data_revisao": agora
                    })

            if linhas_para_salvar:
                with st.spinner("Salvando revisões no Google Sheets..."):
                    try:
                        conn = st.connection("gsheets", type=GSheetsConnection)

                        # Tenta ler o que já existe na planilha
                        try:
                            dados_antigos = conn.read(ttl=0)
                            if dados_antigos is None or dados_antigos.empty:
                                dados_antigos = pd.DataFrame(columns=["arquivo", "trecho", "tags_corrigidas", "data_revisao"])
                        except Exception:
                            dados_antigos = pd.DataFrame(columns=["arquivo", "trecho", "tags_corrigidas", "data_revisao"])

                        novos_dados = pd.DataFrame(linhas_para_salvar)
                        dados_totais = pd.concat([dados_antigos, novos_dados], ignore_index=True)

                        # Envia para a planilha online
                        conn.update(data=dados_totais)
                        st.success(f"✅ {len(linhas_para_salvar)} trecho(s) salvo(s) com sucesso no Google Sheets!")
                    except Exception as e:
                        st.error(f"Erro ao salvar na planilha: {e}")
            else:
                st.warning("Nenhum trecho com tags selecionadas foi marcado para incluir — nada foi salvo.")

    # ----------------------------------------------------------
    # PASSO ⑤ — IMPACTO AI: ANÁLISE INTERPRETATIVA COM IA
    #
    # Cada pesquisador usa a própria chave gratuita da API do
    # Google Gemini (obtida em aistudio.google.com) para gerar um
    # comentário interpretativo sobre os resultados já
    # classificados. Nenhum custo para quem hospeda o programa —
    # cada usuário sustenta seu próprio uso gratuito.
    # ----------------------------------------------------------
    st.divider()
    st.subheader("⑤ Comentário interpretativo com IA (opcional)")
    st.caption(
        "Gera um comentário automático sobre os padrões de impacto "
        "identificados neste relatório, usando o Google Gemini. Requer "
        "sua própria chave de API gratuita — veja a barra lateral."
    )

    chave_gemini = st.session_state.get("gemini_api_key", "")

    if not chave_gemini:
        st.info(
            "🔑 Cole sua chave gratuita da API do Gemini na barra lateral "
            "para habilitar este recurso."
        )
    else:
        if st.button("🧠 Gerar comentário interpretativo", type="primary"):
            with st.spinner("Consultando o Impacto AI (Gemini)..."):
                try:
                    comentario = impacto_ai.gerar_comentario(
                        chave_gemini,
                        resultados_finais,
                        frequencia_tags,
                        st.session_state.get("nome_relatorio", "relatório"),
                    )
                    st.session_state["gemini_comentario"] = comentario
                    st.session_state["gemini_key_validada"] = True
                except impacto_ai.ChaveGeminiInvalida as e:
                    st.error(f"❌ {e}")

        if st.session_state.get("gemini_comentario"):
            st.markdown("**Comentário gerado:**")
            st.info(st.session_state["gemini_comentario"])
            st.caption(
                "⚠️ Este comentário é gerado automaticamente por IA a partir dos "
                "resultados do classificador. Não substitui a revisão humana, "
                "especialmente dos trechos marcados para revisão no passo ④."
            )
