# Guide de déploiement sur Streamlit Cloud

Ce guide vous aidera à déployer votre application de prédiction du revenu annuel marocain sur Streamlit Cloud.

## Prérequis

- Un compte GitHub
- Votre code est déjà poussé sur GitHub (`https://github.com/AbdellahRAISSOUNI/projet-AIML-gi2`)

## Étapes de déploiement

1. Accédez à [Streamlit Cloud](https://streamlit.io/cloud)

2. Connectez-vous avec votre compte GitHub
   
3. Une fois connecté, cliquez sur le bouton "New app" dans le coin supérieur droit

4. Dans la section "Repository", sélectionnez votre dépôt GitHub:
   - Repository: `AbdellahRAISSOUNI/projet-AIML-gi2`

5. Dans la section "Branch", sélectionnez la branche:
   - Branch: `master` (ou `main` si vous utilisez cette branche)

6. Dans la section "Main file path", indiquez le chemin vers votre application Streamlit:
   - Main file path: `app.py`

7. Vous pouvez laisser les "Advanced settings" avec leurs valeurs par défaut

8. Cliquez sur le bouton "Deploy!"

9. Streamlit Cloud va maintenant déployer votre application. Cela peut prendre quelques minutes.

10. Une fois le déploiement terminé, vous recevrez une URL où votre application est accessible

## Gestion de votre application

- Votre application sera automatiquement mise à jour lorsque vous pousserez des changements sur GitHub
- Vous pouvez configurer qui peut accéder à votre application dans les paramètres
- Surveillez l'utilisation des ressources depuis le tableau de bord de Streamlit Cloud

## Résolution des problèmes courants

### Erreur de mémoire

Si votre application rencontre des erreurs de mémoire, envisagez de:

1. Optimiser le chargement des modèles
2. Utiliser `@st.cache_resource` pour charger les modèles une seule fois (déjà implémenté)
3. Réduire la taille des modèles si possible

### Erreur de dépendances

Assurez-vous que toutes les dépendances sont correctement listées dans `requirements.txt` 