"""
Model Training Script for MedIndia's HealthPredict AI
This script trains and saves all disease prediction models using joblib.

Run this script once to generate the .pkl model files.
Usage: python train_models.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
import os

# Create models directory if it doesn't exist
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def train_and_evaluate_models(X_train, X_test, y_train, y_test, disease_name):
    """Train multiple models and return the best one."""

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models to try
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    best_model = None
    best_score = 0
    best_name = ""
    results = {}

    print(f"\n[TRAINING] Training models for {disease_name}...")
    print("-" * 50)

    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

        print(f"  {name}:")
        print(f"    Test Accuracy: {accuracy*100:.2f}%")
        print(f"    CV Score: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

        # Track best model based on CV score
        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean()
            best_model = model
            best_name = name

    print(f"\n[BEST] Best Model: {best_name} (CV: {best_score*100:.2f}%)")

    return best_model, scaler, best_name, results


def train_diabetes_model():
    """Train and save diabetes prediction model."""
    print_header("DIABETES MODEL TRAINING")

    # Load data
    df = pd.read_csv(DATA_DIR / "diabetes.csv")
    print(f"[DATA] Loaded {len(df)} records from diabetes.csv")

    # Preprocess - replace 0s with median for certain columns
    zero_not_valid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_not_valid:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

    # Split features and target
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values
    feature_names = df.drop('Outcome', axis=1).columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train and get best model
    model, scaler, model_name, results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, "Diabetes"
    )

    # Save model and scaler
    joblib.dump(model, MODELS_DIR / "diabetes_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "diabetes_scaler.pkl")
    joblib.dump(feature_names, MODELS_DIR / "diabetes_features.pkl")
    joblib.dump({'model_name': model_name, 'results': results}, MODELS_DIR / "diabetes_info.pkl")

    print(f"[SAVED] diabetes_model.pkl, diabetes_scaler.pkl")

    return model, scaler


def train_heart_model():
    """Train and save heart disease prediction model."""
    print_header("HEART DISEASE MODEL TRAINING")

    # Load data
    df = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")
    print(f"[DATA] Loaded {len(df)} records from Heart_Disease_Prediction.csv")

    # Preprocess
    if 'Heart Disease' in df.columns:
        df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

    df = df.fillna(df.median(numeric_only=True))

    # Split features and target
    X = df.drop('Heart Disease', axis=1).values
    y = df['Heart Disease'].values
    feature_names = df.drop('Heart Disease', axis=1).columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train and get best model
    model, scaler, model_name, results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, "Heart Disease"
    )

    # Save model and scaler
    joblib.dump(model, MODELS_DIR / "heart_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "heart_scaler.pkl")
    joblib.dump(feature_names, MODELS_DIR / "heart_features.pkl")
    joblib.dump({'model_name': model_name, 'results': results}, MODELS_DIR / "heart_info.pkl")

    print(f"[SAVED] heart_model.pkl, heart_scaler.pkl")

    return model, scaler


def train_parkinsons_model():
    """Train and save Parkinson's disease prediction model."""
    print_header("PARKINSON'S DISEASE MODEL TRAINING")

    # Load data
    df = pd.read_csv(DATA_DIR / "parkinsons.csv")
    print(f"[DATA] Loaded {len(df)} records from parkinsons.csv")

    # Preprocess - remove name column
    if 'name' in df.columns:
        df = df.drop('name', axis=1)

    # Split features and target
    X = df.drop('status', axis=1).values
    y = df['status'].values
    feature_names = df.drop('status', axis=1).columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train and get best model
    model, scaler, model_name, results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, "Parkinson's Disease"
    )

    # Save model and scaler
    joblib.dump(model, MODELS_DIR / "parkinsons_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "parkinsons_scaler.pkl")
    joblib.dump(feature_names, MODELS_DIR / "parkinsons_features.pkl")
    joblib.dump({'model_name': model_name, 'results': results}, MODELS_DIR / "parkinsons_info.pkl")

    print(f"[SAVED] parkinsons_model.pkl, parkinsons_scaler.pkl")

    return model, scaler


def train_liver_model():
    """Train and save liver disease prediction model."""
    print_header("LIVER DISEASE MODEL TRAINING")

    # Load data
    df = pd.read_csv(DATA_DIR / "indian_liver_patient.csv")
    print(f"[DATA] Loaded {len(df)} records from indian_liver_patient.csv")

    # Preprocess
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df = df.fillna(df.median(numeric_only=True))
    df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})

    # Split features and target
    X = df.drop('Dataset', axis=1).values
    y = df['Dataset'].values
    feature_names = df.drop('Dataset', axis=1).columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train and get best model
    model, scaler, model_name, results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, "Liver Disease"
    )

    # Save model and scaler
    joblib.dump(model, MODELS_DIR / "liver_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "liver_scaler.pkl")
    joblib.dump(feature_names, MODELS_DIR / "liver_features.pkl")
    joblib.dump({'model_name': model_name, 'results': results}, MODELS_DIR / "liver_info.pkl")

    print(f"[SAVED] liver_model.pkl, liver_scaler.pkl")

    return model, scaler


def train_kidney_model():
    """Train and save chronic kidney disease prediction model."""
    print_header("CHRONIC KIDNEY DISEASE MODEL TRAINING")

    # Load data
    df = pd.read_csv(DATA_DIR / "kidney_disease.csv")
    print(f"[DATA] Loaded {len(df)} records from kidney_disease.csv")

    # Drop id column
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # Encode target
    df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})

    # Encode categorical columns
    categorical_maps = {
        'rbc': {'normal': 1, 'abnormal': 0},
        'pc': {'normal': 1, 'abnormal': 0},
        'pcc': {'present': 1, 'notpresent': 0},
        'ba': {'present': 1, 'notpresent': 0},
        'htn': {'yes': 1, 'no': 0},
        'dm': {'yes': 1, 'no': 0},
        'cad': {'yes': 1, 'no': 0},
        'appet': {'good': 1, 'poor': 0},
        'pe': {'yes': 1, 'no': 0},
        'ane': {'yes': 1, 'no': 0},
    }

    for col, mapping in categorical_maps.items():
        if col in df.columns:
            df[col] = df[col].str.strip().map(mapping)

    # Convert all to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values
    df = df.fillna(df.median(numeric_only=True))

    # Split features and target
    X = df.drop('classification', axis=1).values
    y = df['classification'].values
    feature_names = df.drop('classification', axis=1).columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train and get best model
    model, scaler, model_name, results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, "Chronic Kidney Disease"
    )

    # Save model and scaler
    joblib.dump(model, MODELS_DIR / "kidney_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "kidney_scaler.pkl")
    joblib.dump(feature_names, MODELS_DIR / "kidney_features.pkl")
    joblib.dump({'model_name': model_name, 'results': results}, MODELS_DIR / "kidney_info.pkl")

    print(f"[SAVED] kidney_model.pkl, kidney_scaler.pkl")

    return model, scaler


def train_pancreatic_model():
    """Train and save pancreatic cancer prediction model."""
    print_header("PANCREATIC CANCER MODEL TRAINING")

    # Load data
    df = pd.read_csv(DATA_DIR / "pancreatic_cancer.csv")
    print(f"[DATA] Loaded {len(df)} records from pancreatic_cancer.csv")

    # Drop post-diagnosis columns not available at prediction time
    drop_cols = ['Country', 'Stage_at_Diagnosis', 'Survival_Time_Months', 'Treatment_Type']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode Gender
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Encode ordinal categoricals
    ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}
    for col in ['Physical_Activity_Level', 'Diet_Processed_Food', 'Access_to_Healthcare']:
        if col in df.columns:
            df[col] = df[col].map(ordinal_map)

    if 'Economic_Status' in df.columns:
        df['Economic_Status'] = df['Economic_Status'].map({'Low': 0, 'Middle': 1, 'High': 2})

    # Encode Urban_vs_Rural
    if 'Urban_vs_Rural' in df.columns:
        df['Urban_vs_Rural'] = df['Urban_vs_Rural'].map({'Urban': 1, 'Rural': 0})

    # Split features and target
    X = df.drop('Survival_Status', axis=1).values
    y = df['Survival_Status'].values
    feature_names = df.drop('Survival_Status', axis=1).columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train and get best model
    model, scaler, model_name, results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, "Pancreatic Cancer"
    )

    # Save model and scaler
    joblib.dump(model, MODELS_DIR / "pancreatic_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "pancreatic_scaler.pkl")
    joblib.dump(feature_names, MODELS_DIR / "pancreatic_features.pkl")
    joblib.dump({'model_name': model_name, 'results': results}, MODELS_DIR / "pancreatic_info.pkl")

    print(f"[SAVED] pancreatic_model.pkl, pancreatic_scaler.pkl")

    return model, scaler


def main():
    """Train all models."""
    print("\n" + "=" * 60)
    print("  MedIndia's HealthPredict AI - Model Training")
    print("  Training all disease prediction models...")
    print("=" * 60)

    # Train all models
    train_diabetes_model()
    train_heart_model()
    train_parkinsons_model()
    train_liver_model()
    train_kidney_model()
    train_pancreatic_model()

    # Summary
    print("\n" + "=" * 60)
    print("  [SUCCESS] ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nModels saved in: {MODELS_DIR.absolute()}")
    print("\nFiles created:")
    for f in sorted(MODELS_DIR.glob("*.pkl")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")

    print("\nYou can now run the Streamlit app with pre-trained models!")
    print("   Command: streamlit run app.py")


if __name__ == "__main__":
    main()
