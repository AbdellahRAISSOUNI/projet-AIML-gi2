# API de Prédiction du Revenu Annuel Marocain

Ce document contient les instructions pour utiliser l'API FastAPI pour la prédiction du revenu annuel d'un Marocain.

## Architecture du système

L'application comprend deux composants principaux:

1. **API FastAPI** (api.py): Service backend qui expose le modèle de machine learning via une API REST
2. **Interface Streamlit** (app.py): Interface utilisateur qui interagit avec l'API

## Prérequis

- Python 3.8+
- pip (gestionnaire de paquets Python)
- Dépendances listées dans requirements.txt

## Installation et exécution

### Option 1: Utilisation du script de déploiement local

1. Exécutez le script de déploiement:
   ```bash
   ./deploy_local_with_api.bat  # Sur Windows
   ```
   
   Ce script:
   - Installe les dépendances nécessaires
   - Démarre l'API FastAPI sur le port 8000
   - Démarre l'application Streamlit sur le port 8501

### Option 2: Démarrage manuel des services

1. Installez les dépendances:
   ```bash
   pip install -r requirements.txt
   ```

2. Démarrez l'API FastAPI:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

3. Démarrez l'application Streamlit (dans un autre terminal):
   ```bash
   streamlit run app.py
   ```

## Utilisation de l'API FastAPI

L'API expose plusieurs endpoints:

- **GET /** - Page d'accueil de l'API
- **GET /docs** - Documentation interactive Swagger UI
- **GET /health** - Vérification de l'état de l'API
- **POST /predict** - Prédiction du revenu basée sur les caractéristiques fournies

### Exemple de requête de prédiction

```python
import requests
import json

# URL de l'API
api_url = "http://localhost:8000/predict"

# Données d'entrée pour la prédiction
input_data = {
    "user_id": "USER_TEST",
    "age": 35,
    "categorie_age": "Adulte",
    "date_naissance": "1988-01-01",
    "sexe": "Homme",
    "zone": "Urbain",
    "niveau_education": "Supérieur",
    "annees_experience": 10,
    "etat_matrimonial": "Marié",
    "nombre_enfants": 2,
    "possede_voiture": 1.0,
    "possede_logement": 1.0,
    "possede_terrain": 0.0,
    "categorie_socioprofessionnelle": "Groupe 1: Cadres supérieurs/Directeurs",
    "secteur_activite": "Privé formel",
    "heures_travail_hebdo": 40,
    "couleur_preferee": "Bleu"
}

# Envoi de la requête
response = requests.post(
    api_url,
    json=input_data,
    headers={"Content-Type": "application/json"}
)

# Traitement de la réponse
if response.status_code == 200:
    result = response.json()
    print(f"Revenu annuel prédit: {result['revenu_annuel']:.2f} DH")
    print(f"Revenu mensuel prédit: {result['revenu_mensuel']:.2f} DH")
    print("Contexte:")
    for key, value in result['contexte'].items():
        print(f"  - {key}: {value}")
else:
    print(f"Erreur: {response.status_code} - {response.text}")
```

## Documentation interactive

Une fois l'API démarrée, vous pouvez accéder à la documentation interactive Swagger UI à l'adresse:
http://localhost:8000/docs

Cette documentation vous permet de:
- Explorer les endpoints disponibles
- Tester l'API directement depuis le navigateur
- Voir les schémas de données d'entrée et de sortie

## Intégration avec d'autres applications

L'API peut être intégrée avec n'importe quelle application capable d'envoyer des requêtes HTTP:

- Applications web frontend (React, Vue, etc.)
- Applications mobiles
- Autres services backend
- Outils d'automatisation

## Déploiement en production

Pour déployer l'API en production, voici quelques options recommandées:

1. **Heroku**:
   - Créez un fichier `Procfile` avec le contenu:
     ```
     web: uvicorn api:app --host 0.0.0.0 --port $PORT
     ```
   - Déployez sur Heroku:
     ```
     heroku create
     git push heroku master
     ```

2. **Docker**:
   - Créez un `Dockerfile` pour containeriser l'API
   - Déployez sur des plateformes comme AWS, GCP ou Azure

3. **Serveur dédié ou VPS**:
   - Utilisez nginx comme proxy inverse
   - Configurez un service systemd pour gérer le processus
   - Utilisez Gunicorn comme serveur WSGI 