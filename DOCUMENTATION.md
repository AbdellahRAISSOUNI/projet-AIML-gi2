# Prédiction du Revenu Annuel d'un Marocain

## Table des matières
1. [Introduction](#introduction)
2. [Objectifs du projet](#objectifs-du-projet)
3. [Méthodologie](#méthodologie)
4. [Génération du jeu de données](#génération-du-jeu-de-données)
5. [Analyse exploratoire des données](#analyse-exploratoire-des-données)
6. [Prétraitement des données](#prétraitement-des-données)
7. [Modélisation](#modélisation)
8. [Évaluation des modèles](#évaluation-des-modèles)
9. [Analyse des features importantes](#analyse-des-features-importantes)
10. [Conclusions et perspectives](#conclusions-et-perspectives)

## Introduction

Ce projet vise à prédire le revenu annuel d'un Marocain en se basant sur diverses caractéristiques socio-économiques et démographiques. La prédiction des revenus est un sujet crucial pour la planification économique, l'élaboration de politiques sociales et la compréhension des facteurs qui contribuent aux inégalités de revenus au Maroc.

Dans ce contexte, nous avons développé un modèle de Machine Learning capable d'estimer le revenu annuel d'un individu à partir de données telles que l'âge, le sexe, le niveau d'éducation, la zone géographique (urbaine/rurale), l'expérience professionnelle, et d'autres facteurs socio-économiques pertinents.

## Objectifs du projet

Les principaux objectifs de ce projet sont les suivants :

1. Générer un jeu de données synthétique représentatif de la population marocaine et de ses caractéristiques socio-économiques
2. Effectuer une analyse exploratoire approfondie des données pour comprendre les relations entre les différentes variables
3. Développer et comparer plusieurs modèles de Machine Learning pour prédire le revenu annuel
4. Identifier les facteurs les plus déterminants dans la prédiction du revenu
5. Optimiser le modèle le plus performant pour améliorer sa précision
6. Documenter en détail la méthodologie et les résultats obtenus

## Méthodologie

La méthodologie de ce projet suit les étapes classiques d'un projet de Data Science :

1. **Génération des données** : Création d'un dataset synthétique réaliste basé sur les statistiques officielles marocaines
2. **Exploration et nettoyage des données** : Analyse des distributions, détection et traitement des valeurs manquantes et aberrantes
3. **Prétraitement des données** : Encodage des variables catégorielles, mise à l'échelle des variables numériques
4. **Modélisation** : Développement et entraînement de plusieurs modèles de régression
5. **Évaluation** : Comparaison des performances des différents modèles
6. **Analyse et interprétation** : Identification des variables les plus importantes et interprétation des résultats

## Génération du jeu de données

Pour ce projet, nous avons créé un jeu de données synthétique via le script `generate_dataset.py`. Ce script génère des données qui reflètent la réalité socio-économique du Maroc, en se basant sur les statistiques officielles du Haut-Commissariat au Plan (HCP) et d'autres sources fiables.

Le jeu de données contient 25 359 enregistrements avec 18 variables :

- `user_id` : Identifiant unique pour chaque individu
- `age` : Âge de l'individu (entre 18 et 111 ans)
- `categorie_age` : Catégorie d'âge (Jeune, Adulte, Senior, Âgé)
- `date_naissance` : Date de naissance
- `sexe` : Homme ou Femme
- `zone` : Zone géographique (Urbain ou Rural)
- `niveau_education` : Niveau d'éducation (Sans niveau, Fondamental, Secondaire, Supérieur)
- `annees_experience` : Nombre d'années d'expérience professionnelle
- `etat_matrimonial` : État civil (Célibataire, Marié, Divorcé, Veuf)
- `nombre_enfants` : Nombre d'enfants (0 à 6)
- `possede_voiture` : Possession d'une voiture (0 = Non, 1 = Oui)
- `possede_logement` : Possession d'un logement (0 = Non, 1 = Oui)
- `possede_terrain` : Possession d'un terrain (0 = Non, 1 = Oui)
- `categorie_socioprofessionnelle` : Groupe socioprofessionnel (6 catégories)
- `secteur_activite` : Secteur d'activité (Public, Privé formel, Privé informel, Sans emploi)
- `heures_travail_hebdo` : Nombre d'heures travaillées par semaine
- `couleur_preferee` : Couleur préférée (variable non pertinente)
- `revenu_annuel` : Revenu annuel en Dirhams (variable cible)

La génération du jeu de données a été conçue pour refléter les inégalités de revenus au Maroc, avec des facteurs comme l'écart entre les zones urbaines et rurales, l'écart salarial entre les hommes et les femmes, et l'impact du niveau d'éducation sur les revenus.

## Analyse exploratoire des données

Une analyse exploratoire approfondie a été réalisée dans le notebook `mini_projet_IA_exploration.ipynb`. Cette analyse a permis de comprendre les distributions des variables, leurs corrélations et les tendances générales dans les données.

### Aperçu des données

Le jeu de données compte 25 359 observations et 18 variables, dont une variable cible (`revenu_annuel`). Les données présentent une bonne diversité en termes d'âge, de sexe, de niveau d'éducation et d'autres caractéristiques socio-économiques.

### Statistiques descriptives

L'analyse des statistiques descriptives a révélé plusieurs informations intéressantes :

- L'âge moyen des individus dans notre jeu de données est de 48,7 ans
- Le revenu annuel moyen est de 16 023 DH, avec un écart type important de 13 770 DH
- Le revenu minimum est de 3 000 DH et le maximum est de 63 210 DH
- La moyenne des années d'expérience professionnelle est de 13,3 ans
- Le nombre moyen d'heures travaillées par semaine est de 35,5 heures

### Analyse de la variable cible

La distribution du revenu annuel est asymétrique (skewed), avec une grande majorité des individus ayant des revenus inférieurs à la moyenne. Cette distribution reflète bien les inégalités économiques existantes au Maroc.

### Analyse des relations entre variables

L'étude des corrélations entre les variables a mis en évidence plusieurs relations significatives :

- Corrélation positive entre le niveau d'éducation et le revenu annuel
- Corrélation positive entre les années d'expérience et le revenu
- Écart de revenu significatif entre les zones urbaines et rurales
- Écart de revenu entre les hommes et les femmes
- Impact important de la catégorie socioprofessionnelle sur le revenu

## Prétraitement des données

Avant de procéder à la modélisation, plusieurs étapes de prétraitement ont été appliquées aux données :

1. **Gestion des valeurs manquantes** : Imputation des valeurs manquantes numériques par la médiane et des valeurs catégorielles par le mode
2. **Traitement des valeurs aberrantes** : Identification et traitement des outliers dans les variables numériques
3. **Encodage des variables catégorielles** : Utilisation du One-Hot Encoding pour transformer les variables catégorielles en format numérique
4. **Standardisation des variables numériques** : Mise à l'échelle des variables numériques pour qu'elles aient une moyenne de 0 et un écart-type de 1

Ces étapes de prétraitement ont été intégrées dans un pipeline scikit-learn pour assurer une application cohérente aux données d'entraînement et de test.

## Modélisation

Plusieurs modèles de régression ont été développés et évalués pour prédire le revenu annuel :

1. **Régression linéaire** : Un modèle simple qui établit une relation linéaire entre les variables indépendantes et la variable cible
2. **Arbre de décision** : Un modèle qui capture les relations non linéaires et les interactions entre les variables
3. **Random Forest** : Un ensemble d'arbres de décision qui améliore la performance et la robustesse de la prédiction
4. **Gradient Boosting** : Un algorithme d'ensemble séquentiel qui construit des modèles successifs pour corriger les erreurs des précédents
5. **Réseau de neurones** : Un modèle de deep learning capable de capturer des relations complexes dans les données

Chaque modèle a été entraîné sur un échantillon de 80% des données et évalué sur les 20% restants. Les hyperparamètres ont été optimisés à l'aide de la validation croisée pour améliorer les performances.

## Évaluation des modèles

Les modèles ont été évalués à l'aide de plusieurs métriques :

1. **MAE (Mean Absolute Error)** : L'erreur absolue moyenne entre les prédictions et les valeurs réelles
2. **MSE (Mean Squared Error)** : L'erreur quadratique moyenne, qui pénalise davantage les grandes erreurs
3. **RMSE (Root Mean Squared Error)** : La racine carrée de l'erreur quadratique moyenne, qui est dans la même unité que la variable cible
4. **R² (Coefficient de détermination)** : La proportion de la variance de la variable cible expliquée par le modèle

Les résultats de l'évaluation sont présentés dans un tableau comparatif et visualisés à l'aide de graphiques pour faciliter la comparaison des performances des différents modèles.

## Analyse des features importantes

Une analyse des caractéristiques les plus importantes a été réalisée pour comprendre les facteurs qui influencent le plus le revenu annuel. Cette analyse a été effectuée à l'aide de la méthode de permutation pour le modèle Random Forest, qui permet d'évaluer l'impact de chaque variable sur la précision de la prédiction.

Les résultats de cette analyse sont présentés sous forme de graphique, montrant les 15 variables les plus importantes dans la prédiction du revenu annuel.

## Conclusions et perspectives

### Conclusions principales

1. Les modèles les plus performants pour prédire le revenu annuel sont le Random Forest et le Gradient Boosting, avec des scores R² supérieurs à ceux des autres modèles.
2. Les variables les plus importantes dans la prédiction du revenu sont le niveau d'éducation, la catégorie socioprofessionnelle, le secteur d'activité, la zone géographique et les années d'expérience.
3. L'écart de revenu entre les zones urbaines et rurales, ainsi qu'entre les hommes et les femmes, est significatif et se reflète dans les prédictions du modèle.

### Perspectives d'amélioration

1. **Collecte de données réelles** : Remplacer les données synthétiques par des données réelles pour améliorer la fiabilité du modèle
2. **Features engineering** : Créer de nouvelles variables dérivées qui pourraient capturer des informations plus pertinentes
3. **Modèles plus avancés** : Explorer des architectures de deep learning plus complexes ou des techniques d'ensemble plus sophistiquées
4. **Analyse régionale** : Intégrer des informations géographiques plus précises pour capturer les disparités régionales
5. **Interprétabilité** : Développer des méthodes pour mieux expliquer les prédictions du modèle aux utilisateurs finaux

Ce projet constitue une base solide pour comprendre les facteurs qui influencent le revenu au Maroc et pourrait être étendu pour des applications pratiques dans le domaine de la planification économique et sociale. 