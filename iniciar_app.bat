@echo off
REM ============================================================
REM ABRIR O CLASSIFICADOR DE IMPACTO
REM
REM Basta dar 2 cliques neste arquivo. Uma janela do navegador
REM vai abrir automaticamente com o programa.
REM
REM Para FECHAR o programa: feche esta janela preta (terminal).
REM
REM IMPORTANTE: antes de usar isto pela primeira vez, rode o
REM arquivo "instalar.bat" uma unica vez.
REM ============================================================

cd /d %~dp0

if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo ============================================
    echo  O programa ainda nao foi instalado.
    echo  Rode primeiro o arquivo "instalar.bat"
    echo  ^(duplo clique nele^) e aguarde terminar.
    echo ============================================
    echo.
    pause
    exit /b
)

call .venv\Scripts\activate.bat

echo Abrindo o Classificador de Impacto no navegador...
echo (Nao feche esta janela enquanto estiver usando o programa)
echo.

streamlit run app.py

pause
