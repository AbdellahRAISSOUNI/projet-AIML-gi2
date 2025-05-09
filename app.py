import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Set page configuration
st.set_page_config(
    page_title="Prédiction du Revenu Annuel Marocain",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("🇲🇦 Prédiction du Revenu Annuel d'un Marocain")
st.markdown("""
Cette application permet de prédire le revenu annuel d'un individu au Maroc 
en fonction de ses caractéristiques socio-économiques et démographiques.
""")

# Configuration for API
API_URL = "http://localhost:8000"  # URL when running locally
# For deployed version, we'll use an environment variable or configure based on deployment
if os.environ.get("DEPLOYED") == "true":
    # This will be set in the deployment environment
    API_URL = os.environ.get("API_URL", "https://your-fastapi-deployment-url.com")

# Check if API is available
def check_api_availability():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Function to train a simple model if none exists (fallback if API not available)
@st.cache_resource
def train_simple_model():
    with st.spinner("Aucun modèle trouvé. Entraînement d'un modèle simple en cours..."):
        try:
            # Load the dataset
            df = pd.read_csv('cleaned_dataset_revenu_marocains.csv')
            
            # Separate features and target
            X = df.drop('revenu_annuel', axis=1)
            y = df['revenu_annuel']
            
            # Identify numerical and categorical columns
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            
            # Create preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]), numerical_cols),
                    ('cat', Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ]), categorical_cols)
                ]
            )
            
            # Create a simple RandomForest model
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            # Train the model
            model.fit(X, y)
            
            # Save the model
            joblib.dump(model, 'random_forest_model.pkl')
            
            return model, 'Random Forest'
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de l'entraînement du modèle: {str(e)}")
            return None

# Function to load the best model (fallback if API not available)
@st.cache_resource
def load_model():
    # Check which models are available
    model_files = [f for f in os.listdir() if f.endswith('_model.pkl')]
    
    if not model_files:
        return train_simple_model()
    
    # Default to random forest if available, otherwise use the first model found
    if 'random_forest_model.pkl' in model_files:
        model = joblib.load('random_forest_model.pkl')
        model_name = 'Random Forest'
    else:
        model = joblib.load(model_files[0])
        model_name = model_files[0].replace('_model.pkl', '').replace('_', ' ').title()
    
    return model, model_name

# Load dataset for statistics and feature names
@st.cache_data
def load_data():
    try:
        return pd.read_csv('cleaned_dataset_revenu_marocains.csv')
    except Exception as e:
        st.error(f"Le dataset n'a pas été trouvé: {str(e)}")
        
        # Create a sample dataset for demonstration
        st.info("Création d'un dataset de démonstration...")
        
        # Create a synthetic dataset
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'user_id': [f'USER_{i}' for i in range(n_samples)],
            'age': np.random.randint(18, 80, n_samples),
            'sexe': np.random.choice(['Homme', 'Femme'], n_samples),
            'zone': np.random.choice(['Urbain', 'Rural'], n_samples),
            'niveau_education': np.random.choice(['Sans niveau', 'Fondamental', 'Secondaire', 'Supérieur'], n_samples),
            'annees_experience': np.random.randint(0, 40, n_samples),
            'etat_matrimonial': np.random.choice(['Célibataire', 'Marié', 'Divorcé', 'Veuf'], n_samples),
            'nombre_enfants': np.random.randint(0, 5, n_samples),
            'possede_voiture': np.random.choice([0, 1], n_samples),
            'possede_logement': np.random.choice([0, 1], n_samples),
            'possede_terrain': np.random.choice([0, 1], n_samples),
            'categorie_socioprofessionnelle': np.random.choice([
                'Groupe 1: Cadres supérieurs/Directeurs', 
                'Groupe 2: Cadres moyens/Employés/Commerçants',
                'Groupe 3: Retraités/Rentiers/Inactifs',
                'Groupe 4: Exploitants agricoles/Pêcheurs',
                'Groupe 5: Artisans/Ouvriers qualifiés',
                'Groupe 6: Manœuvres/Petits métiers/Chômeurs'
            ], n_samples),
            'secteur_activite': np.random.choice(['Public', 'Privé formel', 'Privé informel', 'Sans emploi'], n_samples),
            'heures_travail_hebdo': np.random.randint(0, 60, n_samples),
        }
        
        # Generate categorical age
        categorie_age = []
        for age in data['age']:
            if age < 30:
                categorie_age.append("Jeune")
            elif age < 50:
                categorie_age.append("Adulte")
            elif age < 65:
                categorie_age.append("Senior")
            else:
                categorie_age.append("Âgé")
        data['categorie_age'] = categorie_age
        
        # Generate dates
        import datetime
        current_year = datetime.datetime.now().year
        data['date_naissance'] = [f"{current_year - age}-01-01" for age in data['age']]
        
        # Add color preference (non-predictive)
        data['couleur_preferee'] = np.random.choice(['Bleu', 'Rouge', 'Vert', 'Jaune', 'Noir'], n_samples)
        
        # Generate revenue based on features
        base_revenue = 40000
        # Education factor
        edu_factor = {'Sans niveau': 0.7, 'Fondamental': 0.9, 'Secondaire': 1.2, 'Supérieur': 1.5}
        # Experience factor (0.03 per year)
        # Zone factor
        zone_factor = {'Urbain': 1.2, 'Rural': 0.9}
        # Sector factor
        sector_factor = {'Public': 1.1, 'Privé formel': 1.2, 'Privé informel': 0.8, 'Sans emploi': 0.5}
        
        revenue = []
        for i in range(n_samples):
            rev = base_revenue
            rev *= edu_factor[data['niveau_education'][i]]
            rev *= (1 + 0.03 * data['annees_experience'][i])
            rev *= zone_factor[data['zone'][i]]
            rev *= sector_factor[data['secteur_activite'][i]]
            
            # Add random variation
            rev *= np.random.normal(1, 0.2)
            revenue.append(rev)
        
        data['revenu_annuel'] = revenue
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv('cleaned_dataset_revenu_marocains.csv', index=False)
        
        return df

# Function to make predictions via API
def predict_via_api(input_data):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=input_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur de l'API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion à l'API: {str(e)}")
        return None

# Function to make predictions locally (fallback)
def predict_income_locally(model, input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    return {
        "revenu_annuel": float(prediction),
        "revenu_mensuel": float(prediction) / 12,
        "contexte": {
            "smig_mensuel": 2828.71
        }
    }

# Main function
def main():
    # Check if API is available
    is_api_available = check_api_availability()
    
    if is_api_available:
        st.sidebar.success("✅ API connectée")
        st.sidebar.info(f"URL de l'API: {API_URL}")
        model_name = "API FastAPI"
    else:
        st.sidebar.warning("⚠️ API non disponible - Mode local utilisé")
        # Load local model as fallback
        model_tuple = load_model()
        if model_tuple is None:
            return
        model, model_name = model_tuple
    
    # Load data for statistics display
    data = load_data()
    if data is None:
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à", ["Prédiction", "Informations sur le modèle", "Statistiques"])
    
    if page == "Prédiction":
        st.header("Prédiction du Revenu Annuel")
        st.write(f"Modèle utilisé: **{model_name}**")
        
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        with col1:
            # Demographic information
            st.subheader("Informations démographiques")
            
            age = st.slider("Âge", min_value=18, max_value=80, value=35)
            
            if age < 30:
                categorie_age = "Jeune"
            elif age < 50:
                categorie_age = "Adulte"
            elif age < 65:
                categorie_age = "Senior"
            else:
                categorie_age = "Âgé"
            
            sexe = st.selectbox("Sexe", ["Homme", "Femme"])
            zone = st.selectbox("Zone", ["Urbain", "Rural"])
            
            # Standardized date format for consistency
            import datetime
            current_year = datetime.datetime.now().year
            date_naissance = f"{current_year - age}-01-01"
            
            # Family status
            st.subheader("Situation familiale")
            etat_matrimonial = st.selectbox("État matrimonial", ["Célibataire", "Marié", "Divorcé", "Veuf"])
            nombre_enfants = st.slider("Nombre d'enfants", min_value=0, max_value=6, value=1)
        
        with col2:
            # Education and work
            st.subheader("Éducation et travail")
            niveau_education = st.selectbox("Niveau d'éducation", ["Sans niveau", "Fondamental", "Secondaire", "Supérieur"])
            annees_experience = st.slider("Années d'expérience professionnelle", min_value=0, max_value=45, value=10)
            
            categorie_socioprofessionnelle = st.selectbox("Catégorie socioprofessionnelle", [
                'Groupe 1: Cadres supérieurs/Directeurs', 
                'Groupe 2: Cadres moyens/Employés/Commerçants',
                'Groupe 3: Retraités/Rentiers/Inactifs',
                'Groupe 4: Exploitants agricoles/Pêcheurs',
                'Groupe 5: Artisans/Ouvriers qualifiés',
                'Groupe 6: Manœuvres/Petits métiers/Chômeurs'
            ])
            
            secteur_activite = st.selectbox("Secteur d'activité", ["Public", "Privé formel", "Privé informel", "Sans emploi"])
            
            if secteur_activite == "Sans emploi":
                heures_travail_hebdo = 0
            else:
                heures_travail_hebdo = st.slider("Heures de travail hebdomadaires", min_value=0, max_value=80, value=40)
            
            # Assets
            st.subheader("Patrimoine")
            possede_voiture = st.checkbox("Possède une voiture")
            possede_logement = st.checkbox("Possède un logement")
            possede_terrain = st.checkbox("Possède un terrain")
        
        # Other required fields
        user_id = "USER_PREDICT"
        couleur_preferee = "Bleu"  # Non-relevant field for prediction
        
        # Create input data dictionary
        input_data = {
            'user_id': user_id,
            'age': age,
            'categorie_age': categorie_age,
            'date_naissance': date_naissance,
            'sexe': sexe,
            'zone': zone,
            'niveau_education': niveau_education,
            'annees_experience': annees_experience,
            'etat_matrimonial': etat_matrimonial,
            'nombre_enfants': nombre_enfants,
            'possede_voiture': float(possede_voiture),
            'possede_logement': float(possede_logement),
            'possede_terrain': float(possede_terrain),
            'categorie_socioprofessionnelle': categorie_socioprofessionnelle,
            'secteur_activite': secteur_activite,
            'heures_travail_hebdo': heures_travail_hebdo,
            'couleur_preferee': couleur_preferee
        }
        
        # Make prediction when user clicks the button
        if st.button("Prédire le revenu"):
            with st.spinner("Calcul en cours..."):
                try:
                    # Make prediction
                    if is_api_available:
                        # Use API for prediction
                        prediction_result = predict_via_api(input_data)
                    else:
                        # Use local model as fallback
                        prediction_result = predict_income_locally(model, input_data)
                    
                    if prediction_result:
                        prediction = prediction_result["revenu_annuel"]
                        
                        st.success(f"### Revenu annuel prédit: **{prediction:,.2f} DH**")
                        
                        # Show additional context
                        st.info(f"""
                        #### Contexte:
                        - Revenu mensuel estimé: **{prediction/12:,.2f} DH**
                        - Salaire minimum au Maroc (SMIG): ~2,828.71 DH/mois
                        """)
                        
                        # Create gauge chart for visualization
                        fig, ax = plt.subplots(figsize=(10, 2))
                        
                        # Get statistics from the dataset for context
                        min_income = data['revenu_annuel'].min()
                        max_income = data['revenu_annuel'].max()
                        mean_income = data['revenu_annuel'].mean()
                        
                        # Create a gauge-like visualization
                        ax.barh(['Revenu'], [max_income], color='lightgray')
                        ax.barh(['Revenu'], [prediction], color='green')
                        
                        # Add reference lines
                        ax.axvline(x=mean_income, color='blue', linestyle='--', alpha=0.7)
                        ax.text(mean_income, 0, f'Moyenne: {mean_income:,.0f} DH', rotation=90, verticalalignment='bottom')
                        
                        # Format the chart
                        ax.set_xlim(0, max_income * 1.1)
                        ax.set_title('Position du revenu prédit par rapport à la distribution')
                        
                        st.pyplot(fig)
                        
                        # If we have detailed context from API, show it
                        if "contexte" in prediction_result and is_api_available:
                            with st.expander("Détails supplémentaires"):
                                ctx = prediction_result["contexte"]
                                st.write(f"- Revenu minimum dans le dataset: {ctx.get('revenu_min', 0):,.2f} DH")
                                st.write(f"- Revenu maximum dans le dataset: {ctx.get('revenu_max', 0):,.2f} DH")
                                st.write(f"- Revenu moyen dans le dataset: {ctx.get('revenu_moyen', 0):,.2f} DH")
                                st.write(f"- Revenu médian dans le dataset: {ctx.get('revenu_median', 0):,.2f} DH")
                    
                except Exception as e:
                    st.error(f"Une erreur s'est produite lors de la prédiction: {str(e)}")
    
    elif page == "Informations sur le modèle":
        st.header("Informations sur le modèle")
        
        if is_api_available:
            st.subheader("Architecture")
            st.write("""
            Cette application utilise une architecture client-serveur:
            
            1. **API FastAPI**: Un service backend qui héberge le modèle de prédiction
            2. **Interface Streamlit**: Une interface utilisateur qui communique avec l'API
            
            Cette approche offre plusieurs avantages:
            - Séparation des préoccupations (model serving vs interface utilisateur)
            - Possibilité d'utiliser le même modèle pour d'autres applications
            - Meilleure scalabilité et gestion des ressources
            """)
            
            # Simple diagram
            col1, col2, col3 = st.columns([1,3,1])
            with col2:
                st.image("https://mermaid.ink/img/pako:eNpVkE9PwzAMxb9K5HNpcANpoJyQYAdOsMMOu1SukzaC_lG9ZGOo352WtlScnHz-PSVPJ2MdIUGXj4Pjw4FNNpbTkMOyG-0U8_vCrA6Dp2D5S_w7c3RyDdN-5FVnAqfdmqp9d1Tl2I-JazCwJzs4vtXYWXvnMB9Zql5DhpKdhg11r-_yXqJf1wj8-0qSPLMdyhv_XrICsdVeGHFB-lE7GmKs8O9fgV7JDZWRvXVY48t_X3w6ZqjQlXfHrpZniBdDBUuMfHkiKLWpSaBD1qk2okyqMlWNMLXeFGWhpcLpLu-EZZEX1ar5VrdlnWOCJlEKKgVFXQsimoQ6LE6nX8sUczc?type=png", caption="Architecture simplifiée du système")
            
        st.subheader(f"Modèle utilisé: {model_name}")
        
        # Try to load model performance data
        try:
            # Load visuals if they exist
            if os.path.exists('model_comparison.png'):
                st.image('model_comparison.png', caption='Comparaison des performances des modèles')
            else:
                st.info("Les visualisations des modèles ne sont pas disponibles. Elles seront générées automatiquement si vous exécutez model_selection.py.")
            
            if os.path.exists('feature_importance.png'):
                st.image('feature_importance.png', caption='Importance des caractéristiques (Random Forest)')
            
            if os.path.exists('actual_vs_predicted.png'):
                st.image('actual_vs_predicted.png', caption='Valeurs réelles vs prédites')
            
            if os.path.exists('residual_plot.png'):
                st.image('residual_plot.png', caption='Distribution des résidus')
                
        except Exception as e:
            st.warning(f"Impossible de charger certaines visualisations: {str(e)}")
        
        st.subheader("Description des modèles")
        st.markdown("""
        Plusieurs modèles peuvent être entraînés et évalués pour ce projet:
        
        1. **Régression linéaire** : Modèle simple qui établit une relation linéaire entre les variables
        2. **Arbre de décision** : Modèle qui capture les relations non linéaires et les interactions
        3. **Random Forest** : Ensemble d'arbres de décision qui améliore la robustesse
        4. **Gradient Boosting** : Algorithme d'ensemble séquentiel pour améliorer la précision
        5. **Réseau de neurones** : Modèle de deep learning pour capturer des relations complexes
        
        Les modèles sont évalués à l'aide de métriques standard en régression:
        - MAE (Mean Absolute Error) : L'erreur absolue moyenne
        - MSE (Mean Squared Error) : L'erreur quadratique moyenne
        - RMSE (Root Mean Squared Error) : La racine carrée de l'erreur quadratique moyenne
        - R² (Coefficient de détermination) : La proportion de variance expliquée
        """)
    
    elif page == "Statistiques":
        st.header("Statistiques descriptives du dataset")
        
        # Show basic dataset info
        st.subheader("Aperçu du dataset")
        st.write(f"Nombre d'observations: {data.shape[0]}")
        st.write(f"Nombre de colonnes: {data.shape[1]}")
        
        # Display sample data
        st.subheader("Échantillon de données")
        st.dataframe(data.head())
        
        # Display summary statistics
        st.subheader("Statistiques descriptives")
        st.dataframe(data.describe())
        
        # Distribution of target variable
        st.subheader("Distribution des revenus")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data['revenu_annuel'], kde=True, ax=ax)
        ax.set_title('Distribution des revenus annuels')
        ax.set_xlabel('Revenu annuel (DH)')
        ax.set_ylabel('Fréquence')
        st.pyplot(fig)
        
        # Correlation matrix for numerical features
        st.subheader("Matrice de corrélation")
        numerical_data = data.select_dtypes(include=['int64', 'float64'])
        corr = numerical_data.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Matrice de corrélation des variables numériques')
        st.pyplot(fig)
        
        # Distribution of categorical features
        st.subheader("Distribution des variables catégorielles")
        categorical_cols = ['sexe', 'zone', 'niveau_education', 'etat_matrimonial', 'categorie_socioprofessionnelle', 'secteur_activite']
        
        for col in categorical_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=data, x=col, ax=ax)
            ax.set_title(f'Distribution de {col}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Revenu moyen par niveau d'éducation
        st.subheader("Revenu moyen par niveau d'éducation")
        edu_income = data.groupby('niveau_education')['revenu_annuel'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=edu_income, x='niveau_education', y='revenu_annuel', ax=ax)
        ax.set_title("Revenu moyen par niveau d'éducation")
        ax.set_xlabel("Niveau d'éducation")
        ax.set_ylabel("Revenu annuel moyen (DH)")
        st.pyplot(fig)
        
        # Revenu moyen par catégorie socioprofessionnelle
        st.subheader("Revenu moyen par catégorie socioprofessionnelle")
        cat_income = data.groupby('categorie_socioprofessionnelle')['revenu_annuel'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=cat_income, x='categorie_socioprofessionnelle', y='revenu_annuel', ax=ax)
        ax.set_title("Revenu moyen par catégorie socioprofessionnelle")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel("Catégorie socioprofessionnelle")
        ax.set_ylabel("Revenu annuel moyen (DH)")
        plt.tight_layout()
        st.pyplot(fig)

# Run the app
if __name__ == '__main__':
    main() 