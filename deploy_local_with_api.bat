@echo off
echo Démarrage du système de prédiction du revenu annuel...

REM Vérification de Python et pip
python --version
if %ERRORLEVEL% NEQ 0 (
    echo Python n'est pas installé ou n'est pas dans le PATH
    exit /b 1
)

REM Installation des dépendances si nécessaire
echo Installation des dépendances...
pip install -r requirements.txt

REM Mise à jour des dépendances pour FastAPI
pip install fastapi uvicorn

REM Démarrage de l'API FastAPI en arrière-plan
echo Démarrage de l'API FastAPI...
start cmd /k "python api.py"

REM Attendre que l'API soit prête
echo En attente de l'API...
timeout /t 5

REM Démarrage de l'application Streamlit
echo Démarrage de l'application Streamlit...
streamlit run app.py

echo Terminé. 