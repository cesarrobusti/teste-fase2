# 📊 Classificador de Impacto de Pesquisa — Guia de Uso

Este programa lê um relatório (.txt ou .docx) e identifica automaticamente
os trechos com evidências de impacto de pesquisa, classificando cada trecho
com as tags apropriadas.

**Você não precisa saber programar nem entender como o modelo foi treinado
para usar este programa.** Basta seguir os passos abaixo.

---

## 1. O que você recebeu

Confira se os itens abaixo estão presentes na pasta que você recebeu:

```
📁 ClassificadorImpacto/
├── app.py
├── impacto_core.py
├── impacto_ai.py
├── config.py
├── requirements.txt
├── instalar.bat
├── iniciar_app.bat
├── LEIA-ME.md                    ← este arquivo
├── [Excel de quotations de referência]
└── 📁 modelo_impacto/             ← modelo já treinado (não apague nada aqui dentro)
```

Se a pasta `modelo_impacto/` estiver faltando ou vazia, o programa **não vai
funcionar** — avise quem te enviou os arquivos.

Você **não precisa entender** o que cada arquivo faz nem abri-los. Eles
trabalham juntos automaticamente.

---

## 2. Instalação (fazer uma única vez)

### Passo 1 — Instalar o Python
Se o computador ainda não tem Python instalado, baixe a versão 3.10, 3.11 ou
3.12 em **[python.org/downloads](https://www.python.org/downloads/)**.

Durante a instalação, marque a caixinha **"Add python.exe to PATH"** — isso
é importante, sem isso os próximos passos não funcionam.

### Passo 2 — Instalar o programa
Dê **dois cliques em `instalar.bat`**.

Uma janela preta (terminal) vai abrir e instalar tudo sozinha. Isso pode
demorar alguns minutos — é normal, aguarde até aparecer a mensagem
**"Instalação concluída!"**, depois pressione qualquer tecla para fechar.

**Isso só precisa ser feito uma vez**, ao configurar o computador.

---

## 3. Usando o programa (sempre que precisar analisar um relatório)

### Passo 1 — Abrir o programa
Dê **dois cliques em `iniciar_app.bat`**.

Uma janela preta vai abrir (não feche ela — é ela que mantém o programa
rodando) e, em seguida, o programa abre sozinho no seu navegador.

### Passo 2 — Enviar o(s) relatório(s)
Na tela, no passo **①**, escolha:
- **"Enviar arquivo(s)"** para subir um ou **vários** arquivos `.txt`/`.docx`
  de uma só vez (selecione todos juntos na janela de upload), ou
- **"Colar texto diretamente"** para colar o conteúdo de um relatório.

### Passo 3 — Analisar
Clique no botão **"🔍 Analisar relatório(s)"** (passo ②) e aguarde a barra
de progresso terminar. Se você enviou vários arquivos, o programa analisa
todos em sequência.

### Passo 4 — Baixar o resultado
No passo ③, você verá a tabela de resultados na tela, com:
- De qual arquivo veio cada trecho (quando mais de um foi enviado — dá
  para filtrar por arquivo)
- Os trechos identificados
- As tags previstas para cada trecho
- O nível de confiança do modelo
- Quais trechos merecem revisão humana (confiança mais baixa)

Use os botões **"📊 Baixar Excel"** ou **"📄 Baixar CSV"** para salvar os
resultados no seu computador.

### Passo 5 (opcional) — Comentário interpretativo com IA
Se quiser um comentário automático explicando os padrões de impacto
encontrados no relatório, use o passo **⑤** na tela:

1. Na barra lateral (à esquerda), em **"🧠 Impacto AI"**, clique no link
   para obter uma **chave gratuita** em aistudio.google.com (é rápido —
   você só precisa de uma conta Google).
2. Cole a chave no campo indicado.
3. Volte para a tela principal e clique em **"🧠 Gerar comentário
   interpretativo"**.

Esse comentário é gerado por IA a partir dos resultados já classificados —
é um apoio à leitura, não substitui a revisão humana dos trechos marcados
para revisão no passo ③.

Sua chave **não é salva** em nenhum lugar — ela vale só enquanto a aba do
navegador estiver aberta. Da próxima vez que abrir o programa, você
precisa colar a chave de novo (ou pode reutilizar a mesma chave sempre,
sem custo).

### Passo 6 — Fechar o programa
Quando terminar, feche a janela preta (terminal) que ficou aberta.

---

## 4. Perguntas frequentes

**"Posso analisar quantos relatórios eu quiser?"**
Sim, sem limite. Repita os passos 2 a 4 quantas vezes precisar, na mesma
sessão ou em sessões diferentes.

**"Posso analisar vários relatórios de uma vez?"**
Sim — no passo ①, selecione vários arquivos ao mesmo tempo na janela de
upload. Os resultados vêm todos juntos numa mesma tabela, com uma coluna
indicando de qual arquivo veio cada trecho, e você pode filtrar por
arquivo antes de baixar.

**"O programa não abre / dá erro"**
Confira se você fez a instalação (seção 2) primeiro. Se já instalou e mesmo
assim der erro, tire um print da mensagem e envie para quem configurou o
sistema.

**"Preciso estar conectado à internet para usar?"**
Para analisar relatórios com o modelo, não — a análise roda localmente no
seu computador. Internet só é necessária se você usar o recurso opcional
de comentário interpretativo com IA (passo ⑤), que se conecta à API do
Google Gemini.

**"Preciso pagar alguma coisa pelo comentário de IA?"**
Não. Cada pessoa usa sua própria chave gratuita do Google (obtida em
aistudio.google.com). Fica dentro do limite gratuito diário do Google
para uso normal, individual.

**"Posso mover essa pasta para outro lugar do computador?"**
Sim, desde que você mova a pasta **inteira** (incluindo `modelo_impacto/`
e o `.venv` que foi criado na instalação). Se copiar para **outro
computador**, não leve o `.venv` — rode `instalar.bat` de novo lá.

**"Posso enviar um arquivo .doc (Word antigo)?"**
Não diretamente. Abra o arquivo no Word, use "Salvar como" → "Word
Document (.docx)" e envie o `.docx` gerado.

**"O que significa 'trecho para revisão'?"**
São trechos em que o modelo não teve certeza suficiente sobre qual(is)
tag(s) aplicar. Vale a pena um revisor humano conferir esses casos antes de
usar o resultado como definitivo.

**"Posso editar os arquivos `.py`?"**
Não é necessário e não recomendado, a menos que você saiba o que está
fazendo — eles contêm a lógica do programa. Se quiser mudar algum
parâmetro (como o limite de confiança), fale com quem configurou o
sistema.

---

## 5. Precisa de ajuda?

Se algo não funcionar como esperado, anote a mensagem de erro exata
(print da tela ajuda muito) e entre em contato com quem te enviou este
programa.
