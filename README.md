# Prédiction du Revenu Annuel d'un Marocain

Ce projet vise à prédire le revenu annuel d'un Marocain en se basant sur des caractéristiques socio-économiques et démographiques à l'aide de techniques de Machine Learning.

## Structure du projet

```
AI-Project-Pr-diction-du-revenu-annuel-d-un-marocain/
├── mini_projet_IA_exploration.ipynb       # Notebook d'analyse exploratoire
├── generate_dataset.py                    # Script de génération du dataset
├── model_selection.py                     # Script de sélection et d'évaluation des modèles
├── cleaned_dataset_revenu_marocains.csv   # Dataset nettoyé
├── dataset_revenu_marocains.csv           # Dataset brut
├── rapport_analyse_exploratoire.html      # Rapport d'analyse exploratoire
├── DOCUMENTATION.md                       # Documentation détaillée du projet
├── README.md                              # Ce fichier
├── feature_importance.png                 # Visualisation des features importantes
├── model_comparison.png                   # Comparaison des performances des modèles
├── actual_vs_predicted.png                # Visualisation des valeurs réelles vs prédites
└── residual_plot.png                      # Visualisation des résidus du modèle
```

## Prérequis

Pour exécuter ce projet, vous aurez besoin des packages Python suivants :

```
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
```

Vous pouvez les installer avec la commande :

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

## Utilisation

1. **Générer le dataset** (optionnel, déjà inclus dans le dépôt) :

```bash
python generate_dataset.py
```

2. **Explorer les données** :

Ouvrez le notebook Jupyter `mini_projet_IA_exploration.ipynb` pour explorer et visualiser les données.

3. **Entraîner et évaluer les modèles** :

```bash
python model_selection.py
```

Ce script entraîne plusieurs modèles de régression (Régression linéaire, Arbre de décision, Random Forest, Gradient Boosting, et Réseau de neurones), les évalue, et génère des visualisations de comparaison. Les modèles entraînés sont sauvegardés au format pickle.

## Modèles

Les modèles suivants ont été entraînés et évalués :

1. **Régression linéaire** : Un modèle de base pour établir une référence de performance
2. **Arbre de décision** : Capture les relations non linéaires entre les variables
3. **Random Forest** : Ensemble d'arbres de décision pour améliorer la robustesse
4. **Gradient Boosting** : Algorithme d'ensemble séquentiel pour améliorer la précision
5. **Réseau de neurones** : Modèle de deep learning pour capturer des relations complexes

## Résultats

Les performances des modèles sont évaluées à l'aide de métriques standard en régression :
- MAE (Mean Absolute Error) : L'erreur absolue moyenne
- MSE (Mean Squared Error) : L'erreur quadratique moyenne
- RMSE (Root Mean Squared Error) : La racine carrée de MSE
- R² (Coefficient de détermination) : La proportion de variance expliquée

D'après nos évaluations, le modèle Random Forest a obtenu les meilleures performances avec le score R² le plus élevé.

## Documentation détaillée

Pour une documentation détaillée du projet, veuillez consulter le fichier [DOCUMENTATION.md](DOCUMENTATION.md).

## Licence

Ce projet est disponible sous licence MIT.

## Auteurs

- [Nom de l'auteur] 

## Déploiement du Projet

### Option 1: Déploiement sur Streamlit Cloud (Recommandé)

1. Assurez-vous que votre code est déjà poussé sur GitHub (fait avec le script push_to_github.bat)
2. Visitez [Streamlit Cloud](https://streamlit.io/cloud)
3. Connectez-vous avec votre compte GitHub
4. Cliquez sur "New app"
5. Sélectionnez votre dépôt `projet-AIML-gi2`
6. Configurez l'application:
   - Branch: main ou master
   - Main file path: app.py
   - Cliquez sur "Deploy!"

L'application sera déployée et accessible via une URL publique.

### Option 2: Déploiement local

1. Clonez le dépôt:
   ```
   git clone https://github.com/AbdellahRAISSOUNI/projet-AIML-gi2.git
   cd projet-AIML-gi2
   ```

2. Créez un environnement virtuel:
   ```
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. Installez les dépendances:
   ```
   pip install -r requirements.txt
   ```

4. Lancez l'application:
   ```
   streamlit run app.py
   ```

### Option 3: Déploiement sur Heroku

1. Créez un compte sur [Heroku](https://www.heroku.com/)
2. Installez le [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
3. Créez un fichier `Procfile` avec le contenu:
   ```
   web: sh setup.sh && streamlit run app.py
   ```
4. Créez un fichier `setup.sh` avec le contenu:
   ```
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```
5. Ajoutez et poussez vers Heroku:
   ```
   heroku login
   heroku create votre-app-name
   git add .
   git commit -m "Configuration pour Heroku"
   git push heroku master
   ```

### Option 4: Déploiement sur un VPS (Digital Ocean, AWS EC2, etc.)

1. Provisionnez un serveur avec Ubuntu
2. Installez Python et les dépendances
3. Clonez le dépôt
4. Configurez un service systemd ou utilisez [nginx avec gunicorn et streamlit](https://discuss.streamlit.io/t/how-to-use-streamlit-in-docker/1067) 