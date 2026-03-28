import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def load_and_preprocess(data_path='data/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    df = pd.read_csv(data_path)
    df = df.copy()
    
    # Nettoyage
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Encodage des variables catégorielles
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Sauvegarde des encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist(), label_encoders