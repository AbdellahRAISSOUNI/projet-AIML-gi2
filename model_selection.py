import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('cleaned_dataset_revenu_marocains.csv')
print("Dataset loaded successfully with shape:", df.shape)

# Separate features and target
print("\nSeparating features and target...")
X = df.drop('revenu_annuel', axis=1)
y = df['revenu_annuel']

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Create preprocessor
print("\nCreating preprocessor...")
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

# Train-test split
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
print("\nDefining models...")
models = {
    'Linear Regression': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ]),
    'Decision Tree': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', DecisionTreeRegressor(random_state=42))
    ]),
    'Random Forest': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    'Gradient Boosting': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]),
    'Neural Network': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
    ])
}

# Train and evaluate models
print("\nTraining and evaluating models...")
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    })
    
    # Save model
    joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.pkl')

# Create results DataFrame
results_df = pd.DataFrame(results)
print("\nModel Performance Results:")
print(results_df)

# Select best model based on R²
best_model_name = results_df.loc[results_df['R²'].idxmax(), 'Model']
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} with R² = {results_df.loc[results_df['R²'].idxmax(), 'R²']:.4f}")

# Feature importance analysis (for Random Forest)
if 'Random Forest' in models:
    # Create feature importance visualization for Random Forest
    print("\nCalculating feature importance for Random Forest...")
    rf_model = models['Random Forest']
    
    # Get feature names after preprocessing
    preprocessor_rf = rf_model.named_steps['preprocessor']
    preprocessor_rf.fit(X_train)
    
    # Get feature names after one-hot encoding
    cat_features = preprocessor_rf.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([numerical_cols, cat_features])
    
    # Extract the trained RandomForest model
    rf_estimator = rf_model.named_steps['model']
    
    # Calculate permutation importance
    X_train_transformed = preprocessor_rf.transform(X_train)
    
    # Convert sparse matrix to dense array if it's sparse
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    
    perm_importance = permutation_importance(rf_estimator, X_train_transformed, y_train, n_repeats=10, random_state=42)
    
    # Sort features by importance
    sorted_idx = perm_importance.importances_mean.argsort()[-15:]  # Top 15 features
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx])
    plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
    plt.xlabel('Permutation Importance')
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

# Create visualization for model comparison
plt.figure(figsize=(12, 6))
bars = plt.bar(results_df['Model'], results_df['R²'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.title('Model Comparison by R² Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_comparison.png')

# Scatter plot of actual vs predicted values
plt.figure(figsize=(10, 8))
plt.scatter(y_test, best_model.predict(X_test), alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Actual vs Predicted Values ({best_model_name})')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')

# Residual plot
residuals = y_test - best_model.predict(X_test)
plt.figure(figsize=(10, 8))
plt.scatter(best_model.predict(X_test), residuals, alpha=0.5)
plt.hlines(y=0, xmin=min(best_model.predict(X_test)), xmax=max(best_model.predict(X_test)), colors='r', linestyles='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig('residual_plot.png')

print("\nAnalysis complete. Visualizations saved as PNG files.") 