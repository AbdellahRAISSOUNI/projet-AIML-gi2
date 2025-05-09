import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
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

# Function to train a simple model if none exists
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

# Function to load the best model
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

# Function to make predictions
def predict_income(model, input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    return prediction

# Main function
def main():
    # Load the model and data
    model_tuple = load_model()
    data = load_data()
    
    if model_tuple is None or data is None:
        return
    
    model, model_name = model_tuple
    
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
                    prediction = predict_income(model, input_data)
                    
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
                    
                except Exception as e:
                    st.error(f"Une erreur s'est produite lors de la prédiction: {str(e)}")
    
    elif page == "Informations sur le modèle":
        st.header("Informations sur le modèle")
        
        st.subheader(f"Modèle actuel: {model_name}")
        
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