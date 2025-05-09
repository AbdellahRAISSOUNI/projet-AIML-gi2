import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance

print("Loading dataset and model...")
# Load the dataset
df = pd.read_csv('cleaned_dataset_revenu_marocains.csv')
X = df.drop('revenu_annuel', axis=1)
y = df['revenu_annuel']

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Try to load the trained Random Forest model
try:
    print("Loading Random Forest model...")
    rf_model = joblib.load('random_forest_model.pkl')
    
    print("Preparing data for feature importance calculation...")
    # Get the preprocessor from the pipeline
    preprocessor_rf = rf_model.named_steps['preprocessor']
    
    # Transform a small subset of the data to get feature names
    # This reduces memory usage compared to transforming the whole dataset
    X_sample = X.sample(min(1000, len(X)), random_state=42)
    preprocessor_rf.fit(X_sample)
    
    # Get feature names after one-hot encoding
    cat_features = preprocessor_rf.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([numerical_cols, cat_features])
    
    # Extract the trained RandomForest model
    rf_estimator = rf_model.named_steps['model']
    
    print("Creating visualizations for model evaluation...")
    
    # Generate feature importance plots without using permutation importance
    # Use the built-in feature importance of Random Forest instead
    importances = rf_estimator.feature_importances_
    
    # Get indices of the top 15 features
    indices = np.argsort(importances)[-15:]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance visualization saved as 'feature_importance.png'")
    
    # Create visualization for actual vs predicted values
    from sklearn.model_selection import train_test_split
    
    print("Creating actual vs predicted visualization...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = rf_model.predict(X_test)
    
    # Scatter plot of actual vs predicted values
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values (Random Forest)')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    print("Actual vs predicted visualization saved as 'actual_vs_predicted.png'")
    
    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='r', linestyles='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig('residual_plot.png')
    print("Residual plot saved as 'residual_plot.png'")
    
    # Create model comparison visualization
    results = pd.DataFrame([
        {'Model': 'Linear Regression', 'R²': 0.747703},
        {'Model': 'Decision Tree', 'R²': 0.815688},
        {'Model': 'Random Forest', 'R²': 0.886885},
        {'Model': 'Gradient Boosting', 'R²': 0.869277},
        {'Model': 'Neural Network', 'R²': 0.875449}
    ])
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results['Model'], results['R²'], color='skyblue')
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
    print("Model comparison visualization saved as 'model_comparison.png'")
    
    print("All visualizations completed successfully!")
    
except Exception as e:
    print(f"Error: {str(e)}")
    print("Make sure the 'random_forest_model.pkl' file exists and the dataset is available.") 