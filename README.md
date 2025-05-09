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