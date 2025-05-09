from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import uvicorn

app = FastAPI(
    title="API de Prédiction du Revenu Annuel Marocain",
    description="Cette API permet de prédire le revenu annuel d'un individu au Maroc en fonction de ses caractéristiques socio-économiques et démographiques.",
    version="1.0.0"
)

# Define input model
class RevenuInput(BaseModel):
    user_id: str
    age: int
    categorie_age: str
    date_naissance: str
    sexe: str
    zone: str
    niveau_education: str
    annees_experience: int
    etat_matrimonial: str
    nombre_enfants: int
    possede_voiture: float
    possede_logement: float
    possede_terrain: float
    categorie_socioprofessionnelle: str
    secteur_activite: str
    heures_travail_hebdo: int
    couleur_preferee: str

# Define output model
class RevenuOutput(BaseModel):
    revenu_annuel: float
    revenu_mensuel: float
    contexte: dict

# Function to train a model if none exists
def train_model_if_needed():
    # Check if model exists
    model_files = [f for f in os.listdir() if f.endswith('_model.pkl')]
    
    if not model_files:
        print("No model found. Training a new model...")
        
        try:
            # Check if dataset exists
            if not os.path.exists('cleaned_dataset_revenu_marocains.csv'):
                print("Dataset not found. Creating synthetic data...")
                create_synthetic_dataset()
                
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
            print("Model trained and saved successfully.")
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise
    else:
        print(f"Using existing model: {model_files[0]}")

# Function to create synthetic dataset
def create_synthetic_dataset():
    print("Creating synthetic dataset...")
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
    print("Synthetic dataset created and saved successfully.")

# Load the model at startup
@app.on_event("startup")
async def startup_event():
    try:
        train_model_if_needed()
    except Exception as e:
        print(f"Error during startup: {str(e)}")

# Define prediction endpoint
@app.post("/predict", response_model=RevenuOutput)
async def predict(input_data: RevenuInput):
    try:
        # Load the model
        model_files = [f for f in os.listdir() if f.endswith('_model.pkl')]
        if not model_files:
            raise HTTPException(status_code=500, detail="No model found. Please train a model first.")
        
        # Load the best model
        if 'random_forest_model.pkl' in model_files:
            model = joblib.load('random_forest_model.pkl')
        else:
            model = joblib.load(model_files[0])
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get dataset statistics for context
        try:
            df = pd.read_csv('cleaned_dataset_revenu_marocains.csv')
            min_income = df['revenu_annuel'].min()
            max_income = df['revenu_annuel'].max()
            mean_income = df['revenu_annuel'].mean()
            median_income = df['revenu_annuel'].median()
        except:
            # Default values if dataset not available
            min_income = 20000
            max_income = 150000
            mean_income = 60000
            median_income = 55000
        
        # Prepare response
        return RevenuOutput(
            revenu_annuel=float(prediction),
            revenu_mensuel=float(prediction) / 12,
            contexte={
                "revenu_min": float(min_income),
                "revenu_max": float(max_income),
                "revenu_moyen": float(mean_income),
                "revenu_median": float(median_income),
                "smig_mensuel": 2828.71
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "API de Prédiction du Revenu Annuel Marocain",
        "status": "online",
        "endpoints": {
            "/predict": "POST - Faire une prédiction de revenu",
            "/health": "GET - Vérifier l'état de l'API"
        }
    }

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Run the API server when file is executed
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 