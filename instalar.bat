@echo off
REM ============================================================
REM INSTALACAO INICIAL — rodar UMA VEZ apenas.
REM
REM Este arquivo cria o ambiente Python isolado (.venv) e
REM instala tudo que o app precisa para funcionar.
REM
REM Depois de rodar este arquivo com sucesso, use sempre o
REM "iniciar_app.bat" para abrir o programa no dia a dia.
REM ============================================================

cd /d %~dp0

echo ============================================
echo  Instalando o Classificador de Impacto...
echo  Isso pode demorar alguns minutos.
echo ============================================
echo.

python -m venv .venv

call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ============================================
echo  Instalacao concluida!
echo  A partir de agora, use o arquivo
echo  "iniciar_app.bat" para abrir o programa.
echo ============================================
pause
