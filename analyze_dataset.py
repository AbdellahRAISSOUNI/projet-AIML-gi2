import pandas as pd

# Load the dataset with explicit encoding
print("Loading dataset...")
try:
    # Try different encodings
    for encoding in ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']:
        try:
            print(f"Trying encoding: {encoding}")
            df = pd.read_csv('cleaned_dataset_revenu_marocains.csv', encoding=encoding)
            print(f"Successfully loaded with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            print(f"Failed with encoding: {encoding}")
            continue
    
    # Display basic information
    print("\nDataset Shape:", df.shape)
    print("\nColumn Names:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    print("\nFirst 5 rows:\n", df.head())
    print("\nBasic Statistics for numeric columns:\n", df.describe())
    
except Exception as e:
    print("Error loading or processing dataset:", str(e)) 