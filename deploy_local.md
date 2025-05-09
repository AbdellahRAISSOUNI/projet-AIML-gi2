# Guide de déploiement local

Ce guide vous aidera à déployer l'application de prédiction du revenu annuel marocain sur votre machine locale.

## Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Git (pour cloner le dépôt)

## Étapes de déploiement

### 1. Cloner le dépôt

```bash
git clone https://github.com/AbdellahRAISSOUNI/projet-AIML-gi2.git
cd projet-AIML-gi2
```

### 2. Créer un environnement virtuel

#### Sur Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### Sur macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer l'application

```bash
streamlit run app.py
```

Après avoir exécuté cette commande, votre application devrait être accessible à l'adresse http://localhost:8501

## Résolution des problèmes courants

### Problème avec les modèles

Si vous rencontrez des erreurs liées au chargement des modèles:

1. Vérifiez que tous les fichiers `.pkl` sont présents dans le répertoire
2. Si certains modèles sont trop volumineux pour GitHub, vous devrez peut-être les recréer en exécutant:
   ```bash
   python model_selection.py
   ```

### Problème avec les dépendances

Si vous rencontrez des erreurs d'importation ou de dépendances:

1. Assurez-vous que l'environnement virtuel est activé
2. Vérifiez que toutes les dépendances sont installées:
   ```bash
   pip list
   ```
3. Si nécessaire, installez manuellement les dépendances manquantes:
   ```bash
   pip install nom_de_la_dependance
   ```

### Problème d'accès aux fichiers

Si l'application ne trouve pas certains fichiers:

1. Vérifiez que vous exécutez l'application depuis le répertoire racine du projet
2. Vérifiez que les noms de fichiers correspondent exactement (sensible à la casse) 