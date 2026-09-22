# Classificador de Impacto de Pesquisa

Sistema que identifica automaticamente trechos de relatórios com evidências
de impacto de pesquisa e classifica cada trecho com as tags apropriadas,
usando um modelo BERTimbau treinado a partir de quotations já classificadas
manualmente.

Este documento é o guia **técnico**, para quem mantém/desenvolve o projeto.
Se você só vai usar o programa pronto (sem treinar nada), use o
`LEIA-ME.md` em vez deste.

---

## 1. Estrutura de pastas necessária

Todos os arquivos abaixo devem estar **na mesma pasta**:

```
📁 Projeto_Impacto/
│
├── 📄 config.py                     ← configurações (caminhos, thresholds, chaves)
├── 📄 dados_treino.py               ← preparação de dados, compartilhada pelos 3 treinadores
├── 📄 treinar_modelo.py             ← treina o BERTimbau (modelo de produção)
├── 📄 treinar_modelo_setfit.py      ← treina o SetFit (comparação/experimental)
├── 📄 treinar_modelo_tfidf.py       ← treina o baseline TF-IDF + LogReg (comparação)
├── 📄 analisar_relatorio.py         ← analisa 1 relatório, sem tela (terminal)
├── 📄 impacto_core.py               ← funções internas usadas pelo app.py
├── 📄 impacto_ai.py                 ← integração com IA (Gemini) para comentário interpretativo
├── 📄 tags_impacto.py               ← lista completa de tags (edite aqui para adicionar/remover tags)
├── 📄 app.py                        ← programa com tela visual (Streamlit)
├── 📄 requirements.txt              ← lista de dependências do Python
├── 📄 instalar.bat                  ← instala tudo (rodar 1 vez)
├── 📄 iniciar_app.bat               ← abre o programa visual (uso diário)
├── 📄 README.md                     ← este arquivo (guia técnico)
├── 📄 LEIA-ME.md                    ← guia para quem só usa o app pronto
│
├── 📄 Artigo - Impact Tagging-quotations.xlsx   ← Excel de treino original
├── 📄 quotations_revisadas.xlsx                 ← gerado pelo app.py (correções/aprendizado ativo)
├── 📄 [nome do relatório].txt ou .docx          ← relatório a ser analisado
│
├── 📁 modelo_impacto/               ← modelo de PRODUÇÃO (BERTimbau) — criado ao rodar treinar_modelo.py
├── 📁 modelo_impacto_setfit/        ← modelo de comparação (SetFit) — opcional
└── 📁 modelo_impacto_tfidf/         ← modelo de comparação (TF-IDF) — opcional
```

Você **não precisa criar** as pastas `modelo_impacto*/` manualmente — elas
são geradas sozinhas ao rodar o script de treino correspondente.

> **Nota sobre `modelo_impacto_setfit/` e `modelo_impacto_tfidf/`:** esses
> dois só existem para comparar candidatos a modelo. Depois dos testes,
> **o BERTimbau (`modelo_impacto/`) venceu em todas as métricas** e é o
> modelo usado pelo `app.py`/`analisar_relatorio.py`. Os outros dois
> podem ser removidos da versão final que for distribuída — servem só
> para a fase de experimentação.

---

## 2. Para que serve cada arquivo

| Arquivo | O que é | Quando usar |
|---|---|---|
| `config.py` | Guarda todos os caminhos, parâmetros e nomes de modelo | Editar sempre que trocar de relatório ou ajustar parâmetros |
| `dados_treino.py` | Lê, limpa, filtra tags raras e faz o split treino/teste — usado pelos 3 treinadores, garante comparação justa entre eles. Também combina `quotations_revisadas.xlsx` automaticamente | Nunca precisa rodar isso diretamente |
| `treinar_modelo.py` | Treina o **BERTimbau** (modelo de produção) | Rodar na primeira vez, e sempre que quiser incorporar novas quotations ou correções ao modelo |
| `treinar_modelo_setfit.py` | Treina um modelo **SetFit** alternativo, para comparação | Opcional — só para experimentação/comparação |
| `treinar_modelo_tfidf.py` | Treina um baseline **TF-IDF + Regressão Logística**, para comparação | Opcional — só para experimentação/comparação |
| `analisar_relatorio.py` | Analisa **um** relatório usando o modelo já treinado, direto pelo terminal/VSCode (sem tela visual) | Alternativa ao app visual |
| `impacto_core.py` | Funções internas (TF-IDF, classificação, extração de .docx) usadas pelo `app.py` | Nunca precisa abrir isso |
| `impacto_ai.py` | Integração com a API do Google Gemini para gerar comentário interpretativo | Nunca precisa abrir isso |
| `tags_impacto.py` | Lista completa de tags, organizada por bloco, usada no seletor da tela de revisão | **Editar aqui** sempre que uma tag for criada, renomeada ou removida |
| `app.py` | O programa com **tela visual** no navegador | Uso do dia a dia, via `iniciar_app.bat` |
| `requirements.txt` | Lista das bibliotecas Python necessárias | Usado automaticamente por `instalar.bat` |
| `instalar.bat` | Instala o Python virtual e as dependências | Rodar **uma única vez**, ao configurar o computador |
| `iniciar_app.bat` | Abre o app visual no navegador com 2 cliques | Toda vez que for usar o programa |
| `modelo_impacto/` (pasta) | Modelo de produção já treinado (BERTimbau) | Gerada automaticamente; precisa existir para o app funcionar |
| `quotations_revisadas.xlsx` | Correções feitas na tela de revisão do app (aprendizado ativo) | Gerado automaticamente pelo app; entra no próximo treino sem ação manual |

---

## 3. Configuração inicial (fazer uma vez só)

### Passo 1 — Instalar o Python (se ainda não tiver)
Baixe o Python 3.10, 3.11 ou 3.12 em [python.org](https://www.python.org/downloads/)
e instale marcando a opção **"Add python.exe to PATH"** durante a instalação.

### Passo 2 — Instalar as dependências
Dê **duplo clique em `instalar.bat`**. Uma janela preta vai abrir e instalar
tudo sozinha (pode demorar alguns minutos, principalmente por causa do
`torch`, que é um pacote grande). Aguarde até aparecer "Instalação concluída".

### Passo 3 — Treinar o modelo (primeira vez)
Abra o `config.py` e confira se `ARQUIVO_TREINO` está com o nome exato do
seu Excel de quotations classificadas. Depois, no VSCode (ou terminal, com
o ambiente virtual ativado), rode:

```
python treinar_modelo.py
```

Isso vai treinar o BERTimbau e criar a pasta `modelo_impacto/`. Pode demorar
bastante tempo (na base atual, ~2h). **Você só precisa fazer isso de novo
se adicionar novas quotations classificadas ou revisar/corrigir trechos
pelo app** (seção 6).

Os scripts `treinar_modelo_setfit.py` e `treinar_modelo_tfidf.py` são
opcionais — usados apenas para comparar candidatos a modelo. Não são
necessários para o uso normal do sistema.

---

## 4. Uso do dia a dia

Depois da configuração inicial (seção 3), o uso normal é bem simples:

1. Dê **duplo clique em `iniciar_app.bat`**.
2. Uma janela do navegador abre automaticamente com o programa.
3. Siga os passos numerados na tela: ① envie o relatório (.txt ou .docx) →
   ② clique em "Analisar" → ③ baixe o resultado em Excel ou CSV.
4. Para fechar o programa, feche a janela preta (terminal) que abriu junto.

**Não precisa mexer em código nenhum para o uso diário.**

### Alternativa sem tela visual
Se preferir rodar pelo terminal/VSCode em vez do app visual: edite
`NOVO_RELATORIO` em `config.py` com o nome do arquivo, depois rode:

```
python analisar_relatorio.py
```

O resultado é salvo direto no arquivo definido em `ARQUIVO_RESULTADO`
(também em `config.py`).

---

## 5. Comentário interpretativo com IA (Impacto AI)

O app inclui um recurso opcional (passo ⑤ na tela), chamado **Impacto AI**,
que gera um comentário interpretativo automático sobre os resultados de um
relatório, usando a API do **Google Gemini**.

**Cada pesquisador usa a própria chave gratuita**, obtida em
[aistudio.google.com](https://aistudio.google.com/apikey). Isso significa:

- **Nenhum custo para quem hospeda/distribui o programa** — cada pessoa
  sustenta seu próprio uso da cota gratuita do Google.
- A chave é colada na barra lateral do app e fica **apenas na sessão do
  navegador** (nunca é salva em disco por este programa).
- Sem chave, o app funciona normalmente — esse recurso é 100% opcional.

O modelo usado é definido em `config.py` (`GEMINI_MODEL_NAME`), atualmente
`gemini-3.5-flash-lite` (camada gratuita do Google). **Atenção:** o Google
descontinua modelos com alguma frequência — se parar de funcionar com erro
"model ... is no longer available", troque `GEMINI_MODEL_NAME` pelo nome
indicado na própria mensagem de erro, ou confira a lista atual em
[ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models).
A integração usa o SDK novo do Google (`google-genai`) — se atualizar
manualmente, não confunda com o pacote antigo `google-generativeai`,
que foi descontinuado.

---

## 6. Revisão e aprendizado ativo

No app, depois de analisar um relatório, a seção ④ ("Revisar, corrigir e
ensinar o modelo") permite corrigir manualmente as tags previstas —
especialmente útil nos trechos marcados para revisão (confiança baixa).

Essas correções são salvas em `quotations_revisadas.xlsx`. Da próxima vez
que você rodar `treinar_modelo.py`, o `dados_treino.py` combina
automaticamente esse arquivo com o Excel de treino original — sem
precisar copiar nada manualmente. Se o mesmo trecho aparecer nos dois
arquivos, a versão corrigida por você prevalece.

> ⚠️ **Nota sobre o status atual deste recurso:** por enquanto, o fluxo
> de revisão/retreino está ativo apenas para fins de teste e validação do
> pipeline (rodando localmente). Na versão final que for distribuída para
> os pesquisadores, essa seção pode ser removida da interface — avalie
> se faz sentido manter, dependendo de quem terá acesso ao retreino.

---

## 7. Comparando os 3 classificadores (BERTimbau, SetFit, TF-IDF)

Os três scripts de treino (`treinar_modelo.py`, `treinar_modelo_setfit.py`,
`treinar_modelo_tfidf.py`) usam o **mesmo** conjunto de treino/teste (via
`dados_treino.py`), então as métricas impressas ao final
(`eval_f1_macro`, `eval_f1_micro`, `eval_f1_weighted`) são diretamente
comparáveis entre eles.

Resultado da última comparação feita neste projeto:

| Modelo | F1 macro | F1 micro | F1 weighted | Tempo de treino |
|---|---|---|---|---|
| **BERTimbau** (produção) | **0,363** | **0,560** | **0,532** | ~2h |
| SetFit | 0,328 | 0,531 | 0,438 | ~8h |
| TF-IDF + LogReg | 0,146 | 0,367 | 0,311 | segundos |

BERTimbau venceu em todas as métricas e ainda foi mais rápido que o
SetFit nesta base — por isso é o modelo usado em produção.

---

## 8. Compartilhando com outros pesquisadores

Para a versão final de distribuição, veja o `LEIA-ME.md` — ele já lista
exatamente o subconjunto de arquivos necessário para quem só vai usar o
programa pronto (sem os scripts de treino/comparação).

---

## 9. Perguntas frequentes

**"Já treinei o modelo, preciso rodar `treinar_modelo.py` de novo todo dia?"**
Não. Treine uma vez, e use `app.py` (ou `analisar_relatorio.py`) para
analisar quantos relatórios quiser. Retreine se adicionar novas
quotations classificadas, ou se acumular correções pelo app (seção 6).

**"O app não abre / diz que não encontrou o modelo"**
Confira se a pasta `modelo_impacto/` existe e está na mesma pasta do
`app.py`. Se não existir, rode `treinar_modelo.py` primeiro.

**"Posso analisar um arquivo .doc antigo do Word?"**
Não diretamente — abra o arquivo no Word e use "Salvar como" → "Word
Document (.docx)", depois envie o `.docx` gerado.

**"Onde ajusto o quanto o modelo é 'rigoroso' para aceitar uma tag?"**
No app visual, na barra lateral, em "Parâmetros avançados". No
`analisar_relatorio.py`, edite `SIMILARITY_THRESHOLD` e
`CLASSIFICATION_THRESHOLD` em `config.py`.

**"O comentário de IA é confiável?"**
É um apoio interpretativo, não uma conclusão definitiva — sempre trate
como um rascunho a ser revisado, principalmente porque ele é gerado só a
partir do resumo estruturado dos resultados, não do relatório completo.
