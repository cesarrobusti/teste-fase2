# 📊 Classificador de Impacto de Pesquisa

Sistema online baseado em Processamento de Linguagem Natural (PLN) para identificar e classificar automaticamente evidências de impacto científico, tecnológico e socioeconômico em relatórios técnicos e acadêmicos.

O classificador utiliza um modelo de linguagem **BERTimbau** finetunado para o domínio de impacto a partir de trechos anotados manualmente (*quotations*).

---

## 🚀 Acesso Rápido (Sem Instalação)

O aplicativo está disponível diretamente no navegador, pronto para uso em qualquer computador (Windows, macOS ou Linux):

👉 **[Acessar o Classificador de Impacto](https://testeimpacto.streamlit.app/)**

> **Não é necessário instalar Python, Git ou bibliotecas no seu computador.** Toda a infraestrutura roda na nuvem via Streamlit Community Cloud, com download dinâmico dos pesos via Hugging Face Hub.

---

## 📖 Como Usar o Aplicativo

### Passo ① — Enviar o(s) relatório(s)
Você pode analisar relatórios de duas maneiras:
* **📎 Enviar arquivo(s):** Selecione um ou vários arquivos em formato `.txt` ou `.docx` (Word). *Caso possua um arquivo antigo `.doc`, abra no Word e salve como `.docx` antes de enviar.*
* **📋 Colar texto diretamente:** Cole trechos de textos ou minutas para análise rápida no formulário.

### Passo ② — Executar a Análise
Clique em **"🔍 Analisar relatório(s)"**. O sistema executa o pipeline em duas etapas:
1. **Filtro de Relevância (TF-IDF):** Segmenta o texto em parágrafos e seleciona apenas os trechos com similaridade semântica com evidências humanas de referência.
2. **Classificação Multilabel (BERTimbau):** O modelo infere as probabilidades de cada uma das tags de impacto sobre os trechos pré-selecionados.

### Passo ③ — Visualizar e Baixar os Resultados
* Os resultados são exibidos em uma tabela dinâmica com os trechos extraídos, as tags atribuídas, o percentual de confiança e o marcador de **revisão humana** (trechos com confiança limítrofe).
* Baixe os relatórios estruturados nos formatos:
  * **📊 Excel (.xlsx):** Inclui abas separadas para os trechos classificados e para o sumário de frequência das tags.
  * **📄 CSV (.csv):** Formato tabular leve compatível com softwares estatísticos e bancos de dados.

### Passo ④ — Revisão e Aprendizado Ativo *(Opcional)*
Permite que o pesquisador corrija manualmente previsões incorretas ou trechos marcados para revisão humana. As correções podem ser salvas para compor futuras rodadas de retreinamento do modelo.

### Passo ⑤ — Impacto AI: Análise Interpretativa *(Opcional)*
Gera um diagnóstico textual interpretativo dos padrões de impacto encontrados, sintetizado por Inteligência Artificial generativa:
1. Obtenha uma chave gratuita da API do Google Gemini em [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Cole sua chave no campo **"Sua chave de API do Gemini"** na barra lateral do app e clique em **"Validar chave"**.
3. Clique em **"🧠 Gerar comentário interpretativo"** para produzir o resumo.

*A chave é individual, não tem custo e fica restrita apenas à sessão atual do seu navegador (não é salva em banco de dados nem no servidor).*

---

## 🛠️ Arquitetura e Estrutura Técnica

Para pesquisadores e desenvolvedores que desejam inspecionar o pipeline ou rodar experimentos:

### 1. Desacoplamento Nuvem
* **Código e Interface:** Hospedados neste repositório GitHub (`cesarrobusti/teste-fase2`).
* **Pesos do Modelo (~400 MB):** Hospedados publicamente no Hugging Face Hub (`cesarrobusti/teste`).
* **Deploy e Execução:** Gerenciados pelo Streamlit Community Cloud com cacheamento em memória (`@st.cache_resource`).

### 2. Principais Componentes
* `app.py`: Interface de usuário, formulários e renderização de tabelas (Streamlit).
* `impacto_core.py`: Lógica do pipeline de inferência, extração de texto de arquivos `.docx`/`.txt`, vetorização TF-IDF e inferência PyTorch.
* `config.py`: Parâmetros do modelo, limiares de confiança (*thresholds*) e caminhos de referência.
* `impacto_ai.py`: Integração com a API do Google Gemini para síntese qualitativa.
* `tags_impacto.py`: Dicionário e categorização das tags de impacto aceitas pelo classificador.

### 3. Execução Local para Desenvolvimento (Opcional)
Caso queira modificar o código localmente:

```bash
# 1. Clone o repositório
git clone [https://github.com/cesarrobusti/teste-fase2.git](https://github.com/cesarrobusti/teste-fase2.git)
cd teste-fase2

# 2. Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate   # No Linux/macOS
.venv\Scripts\activate      # No Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Inicie o app localmente
streamlit run app.py
