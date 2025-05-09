import numpy as np
import pandas as pd
from datetime import datetime
import random

# Configuration du générateur de nombres aléatoires pour la reproductibilité
np.random.seed(42)
random.seed(42)

# Nombre d'enregistrements à générer
n_samples = 40000

def generate_dataset():
    """
    Génère un dataset synthétique réaliste pour la prédiction du revenu annuel des Marocains
    selon les contraintes statistiques du HCP.
    """
    # Définition des paramètres de distribution selon les contraintes
    
    # 1. Zone (urbain/rural)
    # Répartition approximative: 60% urbain, 40% rural
    zone = np.random.choice(['Urbain', 'Rural'], size=n_samples, p=[0.6, 0.4])
    
    # 2. Âge (entre 18 et 80 ans)
    age = np.random.randint(18, 81, size=n_samples)
    
    # 3. Catégorie d'âge
    categorie_age = []
    for a in age:
        if a < 30:
            categorie_age.append('Jeune')
        elif a < 50:
            categorie_age.append('Adulte')
        elif a < 65:
            categorie_age.append('Senior')
        else:
            categorie_age.append('Âgé')
    
    # 4. Sexe (environ 50/50)
    sexe = np.random.choice(['Homme', 'Femme'], size=n_samples)
    
    # 5. Niveau d'éducation (sans niveau, fondamental, secondaire, supérieur)
    # Répartition approximative basée sur les statistiques marocaines
    niveaux_education = ['Sans niveau', 'Fondamental', 'Secondaire', 'Supérieur']
    proba_education = [0.3, 0.4, 0.2, 0.1]
    niveau_education = np.random.choice(niveaux_education, size=n_samples, p=proba_education)
    
    # 6. Années d'expérience (0 à age-18 ans avec maximum 45)
    annees_experience = []
    for a in age:
        max_exp = min(a - 18, 45)  # Maximum 45 ans d'expérience
        if max_exp < 0:
            max_exp = 0
        annees_experience.append(np.random.randint(0, max_exp + 1))
    
    # 7. État matrimonial
    etats_matrimoniaux = ['Célibataire', 'Marié', 'Divorcé', 'Veuf']
    proba_etat = []
    etat_matrimonial = []
    
    for a in age:
        if a < 25:
            proba_etat = [0.9, 0.1, 0.0, 0.0]
        elif a < 35:
            proba_etat = [0.5, 0.45, 0.05, 0.0]
        elif a < 50:
            proba_etat = [0.2, 0.65, 0.1, 0.05]
        else:
            proba_etat = [0.1, 0.6, 0.1, 0.2]
        etat_matrimonial.append(np.random.choice(etats_matrimoniaux, p=proba_etat))
    
    # 8. Possession de biens
    possede_voiture = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    possede_logement = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    possede_terrain = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    
    # 9. Catégorie socioprofessionnelle (6 groupes)
    categories_socio = [
        'Groupe 1: Cadres supérieurs/Directeurs', 
        'Groupe 2: Cadres moyens/Employés/Commerçants',
        'Groupe 3: Retraités/Rentiers/Inactifs',
        'Groupe 4: Exploitants agricoles/Pêcheurs',
        'Groupe 5: Artisans/Ouvriers qualifiés',
        'Groupe 6: Manœuvres/Petits métiers/Chômeurs'
    ]
    
    # Attribution des catégories socioprofessionnelles en fonction du niveau d'éducation et de l'âge
    categorie_socio = []
    for i in range(n_samples):
        edu = niveau_education[i]
        exp = annees_experience[i]
        
        if edu == 'Supérieur' and exp > 10:
            proba_cat = [0.5, 0.4, 0.05, 0.0, 0.05, 0.0]
        elif edu == 'Supérieur':
            proba_cat = [0.3, 0.5, 0.05, 0.0, 0.1, 0.05]
        elif edu == 'Secondaire' and exp > 15:
            proba_cat = [0.1, 0.4, 0.1, 0.05, 0.3, 0.05]
        elif edu == 'Secondaire':
            proba_cat = [0.05, 0.3, 0.1, 0.1, 0.35, 0.1]
        elif edu == 'Fondamental' and exp > 20:
            proba_cat = [0.0, 0.15, 0.15, 0.2, 0.4, 0.1]
        elif edu == 'Fondamental':
            proba_cat = [0.0, 0.1, 0.1, 0.2, 0.3, 0.3]
        else:  # Sans niveau
            proba_cat = [0.0, 0.0, 0.2, 0.3, 0.2, 0.3]
            
        # Ajuster pour les retraités
        if age[i] >= 60:
            proba_cat[2] += 0.4  # Augmenter la probabilité du Groupe 3 (retraités)
            total = sum(proba_cat)
            proba_cat = [p/total for p in proba_cat]  # Normaliser
            
        categorie_socio.append(np.random.choice(categories_socio, p=proba_cat))
    
    # 10. Variables supplémentaires
    
    # a. Nombre d'enfants (0 à 6)
    # Dépend de l'âge et de l'état matrimonial
    nombre_enfants = []
    for i in range(n_samples):
        a = age[i]
        etat = etat_matrimonial[i]
        
        if etat == 'Célibataire':
            proba_enfants = [0.9, 0.07, 0.02, 0.01, 0.0, 0.0, 0.0]  # 0 à 6 enfants
        elif etat == 'Marié':
            if a < 30:
                proba_enfants = [0.5, 0.3, 0.15, 0.05, 0.0, 0.0, 0.0]
            elif a < 45:
                proba_enfants = [0.1, 0.2, 0.3, 0.2, 0.1, 0.07, 0.03]
            else:
                proba_enfants = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
        else:  # Divorcé ou Veuf
            proba_enfants = [0.2, 0.3, 0.25, 0.15, 0.05, 0.03, 0.02]
            
        nombre_enfants.append(np.random.choice(range(7), p=proba_enfants))
    
    # b. Secteur d'activité
    secteurs = ['Public', 'Privé formel', 'Privé informel', 'Sans emploi']
    
    # Probabilité en fonction de la catégorie socioprofessionnelle
    secteur_activite = []
    for cat in categorie_socio:
        if 'Groupe 1' in cat:
            proba_secteur = [0.6, 0.35, 0.0, 0.05]
        elif 'Groupe 2' in cat:
            proba_secteur = [0.4, 0.5, 0.05, 0.05]
        elif 'Groupe 3' in cat:
            proba_secteur = [0.1, 0.1, 0.2, 0.6]
        elif 'Groupe 4' in cat:
            proba_secteur = [0.05, 0.15, 0.7, 0.1]
        elif 'Groupe 5' in cat:
            proba_secteur = [0.1, 0.4, 0.4, 0.1]
        else:  # Groupe 6
            proba_secteur = [0.05, 0.15, 0.6, 0.2]
            
        secteur_activite.append(np.random.choice(secteurs, p=proba_secteur))
    
    # c. Nombre d'heures travaillées par semaine
    heures_travail = []
    for secteur in secteur_activite:
        if secteur == 'Sans emploi':
            heures_travail.append(0)
        elif secteur == 'Public':
            heures_travail.append(np.random.normal(40, 5))
        elif secteur == 'Privé formel':
            heures_travail.append(np.random.normal(45, 8))
        else:  # Privé informel
            heures_travail.append(np.random.normal(50, 15))
    
    # Assurer que les heures sont positives et plafonner à 80 heures
    heures_travail = [max(0, min(int(h), 80)) for h in heures_travail]
    
    # 11. Colonnes redondantes ou non pertinentes
    # Date de naissance (redondante avec l'âge)
    annee_actuelle = datetime.now().year
    date_naissance = [f"{annee_actuelle - a}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}" for a in age]
    
    # Couleur préférée (non pertinente)
    couleurs = ['Rouge', 'Bleu', 'Vert', 'Jaune', 'Noir', 'Blanc', 'Orange', 'Violet']
    couleur_preferee = np.random.choice(couleurs, size=n_samples)
    
    # ID utilisateur (non pertinent)
    user_id = [f"USER_{i:05d}" for i in range(n_samples)]
    
    # 12. Génération du revenu annuel selon les contraintes
    # Revenu moyen global: 21.949 DH/an
    # Urbain: 26.988 DH, Rural: 12.862 DH
    # 71,8% des revenus < moyenne (urbain 65,9%, rural 85,4%)
    
    revenu = []
    
    for i in range(n_samples):
        # Facteurs de base selon la zone
        if zone[i] == 'Urbain':
            base_mean = 26988
            base_std = 15000
        else:  # Rural
            base_mean = 12862
            base_std = 8000
        
        # Facteurs d'ajustement (multiplicatifs)
        # Âge et expérience
        age_factor = 0.5 + min(annees_experience[i] / 30, 1.5)
        
        # Niveau d'éducation
        education_factors = {
            'Sans niveau': 0.7,
            'Fondamental': 0.9,
            'Secondaire': 1.2,
            'Supérieur': 1.8
        }
        edu_factor = education_factors[niveau_education[i]]
        
        # Sexe (écart salarial)
        sexe_factor = 1.2 if sexe[i] == 'Homme' else 0.9
        
        # Catégorie socioprofessionnelle
        socio_factors = {
            'Groupe 1: Cadres supérieurs/Directeurs': 2.5,
            'Groupe 2: Cadres moyens/Employés/Commerçants': 1.8,
            'Groupe 3: Retraités/Rentiers/Inactifs': 1.0,
            'Groupe 4: Exploitants agricoles/Pêcheurs': 0.8,
            'Groupe 5: Artisans/Ouvriers qualifiés': 1.1,
            'Groupe 6: Manœuvres/Petits métiers/Chômeurs': 0.6
        }
        socio_factor = socio_factors[categorie_socio[i]]
        
        # Secteur d'activité
        secteur_factors = {
            'Public': 1.3,
            'Privé formel': 1.2,
            'Privé informel': 0.8,
            'Sans emploi': 0.3
        }
        secteur_factor = secteur_factors[secteur_activite[i]]
        
        # Calcul du revenu avec bruit gaussien
        revenu_mean = base_mean * age_factor * edu_factor * sexe_factor * socio_factor * secteur_factor
        
        # Ajout de bruit aléatoire
        revenu_i = np.random.normal(revenu_mean, revenu_mean * 0.2)
        
        # S'assurer que le revenu est positif
        revenu_i = max(3000, revenu_i)  # Minimum 3000 DH/an
        
        revenu.append(int(revenu_i))
    
    # 13. Insertion de valeurs manquantes et aberrantes
    
    # Convertir en DataFrame
    df = pd.DataFrame({
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
        'possede_voiture': possede_voiture,
        'possede_logement': possede_logement,
        'possede_terrain': possede_terrain,
        'categorie_socioprofessionnelle': categorie_socio,
        'secteur_activite': secteur_activite,
        'heures_travail_hebdo': heures_travail,
        'couleur_preferee': couleur_preferee,
        'revenu_annuel': revenu
    })
    
    # Insertion de valeurs manquantes (environ 5%)
    for col in ['annees_experience', 'niveau_education', 'etat_matrimonial', 
                'possede_voiture', 'possede_logement', 'possede_terrain', 
                'secteur_activite', 'heures_travail_hebdo']:
        mask = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
        df.loc[mask, col] = np.nan
    
    # Insertion de valeurs aberrantes (~1%)
    # Âge aberrant (>100 ans)
    mask_age = np.random.choice([True, False], size=n_samples, p=[0.01, 0.99])
    df.loc[mask_age, 'age'] = np.random.randint(100, 150, size=sum(mask_age))
    
    # Experience aberrante (>âge-18)
    mask_exp = np.random.choice([True, False], size=n_samples, p=[0.01, 0.99])
    df.loc[mask_exp, 'annees_experience'] = df.loc[mask_exp, 'age'] + np.random.randint(1, 10, size=sum(mask_exp))
    
    # Revenus aberrants (très élevés)
    mask_rev = np.random.choice([True, False], size=n_samples, p=[0.01, 0.99])
    df.loc[mask_rev, 'revenu_annuel'] = np.random.randint(1000000, 5000000, size=sum(mask_rev))
    
    # Heures de travail aberrantes
    mask_hrs = np.random.choice([True, False], size=n_samples, p=[0.01, 0.99])
    df.loc[mask_hrs, 'heures_travail_hebdo'] = np.random.randint(100, 168, size=sum(mask_hrs))
    
    return df

if __name__ == "__main__":
    # Générer le dataset
    df = generate_dataset()
    
    # Vérifier quelques statistiques clés
    total_mean = df['revenu_annuel'].mean()
    urban_mean = df[df['zone'] == 'Urbain']['revenu_annuel'].mean()
    rural_mean = df[df['zone'] == 'Rural']['revenu_annuel'].mean()
    
    print(f"Revenu moyen global: {total_mean:.2f} DH/an")
    print(f"Revenu moyen urbain: {urban_mean:.2f} DH/an")
    print(f"Revenu moyen rural: {rural_mean:.2f} DH/an")
    
    # Pourcentage de revenus inférieurs à la moyenne
    below_mean_total = (df['revenu_annuel'] < total_mean).mean() * 100
    below_mean_urban = (df[df['zone'] == 'Urbain']['revenu_annuel'] < urban_mean).mean() * 100
    below_mean_rural = (df[df['zone'] == 'Rural']['revenu_annuel'] < rural_mean).mean() * 100
    
    print(f"Pourcentage des revenus < moyenne (global): {below_mean_total:.1f}%")
    print(f"Pourcentage des revenus < moyenne (urbain): {below_mean_urban:.1f}%")
    print(f"Pourcentage des revenus < moyenne (rural): {below_mean_rural:.1f}%")
    
    # Enregistrer le dataset dans un fichier CSV
    df.to_csv('dataset_revenu_marocains.csv', index=False)
    print(f"\nDataset généré et sauvegardé dans 'dataset_revenu_marocains.csv'")
    print(f"Nombre d'enregistrements: {len(df)}")