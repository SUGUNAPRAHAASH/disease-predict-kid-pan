"""
Flask REST API for Multi-Disease Prediction System
HealthPredict AI by MedIndia

This API provides endpoints for:
- Diabetes Risk Assessment
- Heart Disease Check
- Parkinson's Screening
- Liver Health Analysis
- Chronic Kidney Disease Assessment
- Pancreatic Cancer Screening
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

app = Flask(__name__)
CORS(app)  # Enable CORS for ASP.NET Core frontend

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Cache for loaded models
_model_cache = {}


def get_model_path(disease_type, file_type):
    """Get path for model files."""
    return os.path.join(MODELS_DIR, f'{disease_type}_{file_type}.pkl')


def load_model(disease_type):
    """Load pre-trained model and scaler from disk."""
    if disease_type in _model_cache:
        return _model_cache[disease_type]

    try:
        model = joblib.load(get_model_path(disease_type, 'model'))
        scaler = joblib.load(get_model_path(disease_type, 'scaler'))
        features = joblib.load(get_model_path(disease_type, 'features'))
        info = joblib.load(get_model_path(disease_type, 'info'))

        _model_cache[disease_type] = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'info': info
        }
        return _model_cache[disease_type]
    except Exception as e:
        print(f"Error loading model for {disease_type}: {e}")
        # Train model if not found
        return train_model_fallback(disease_type)


def load_diabetes_data():
    """Load and preprocess diabetes dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'diabetes.csv'))

    # Replace zeros with median for certain columns
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        if col in df.columns:
            median_val = df[df[col] != 0][col].median()
            df[col] = df[col].replace(0, median_val)

    return df


def load_heart_data():
    """Load and preprocess heart disease dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'Heart_Disease_Prediction.csv'))

    # Map target column if needed
    if 'Heart Disease' in df.columns:
        df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
        if df['Heart Disease'].isna().any():
            df['Heart Disease'] = pd.to_numeric(df['Heart Disease'], errors='coerce')

    return df


def load_parkinsons_data():
    """Load and preprocess Parkinson's dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'parkinsons.csv'))

    # Remove name column if present
    if 'name' in df.columns:
        df = df.drop('name', axis=1)

    return df


def load_liver_data():
    """Load and preprocess liver disease dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'indian_liver_patient.csv'))

    # Encode gender
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Fill missing values
    df = df.fillna(df.median(numeric_only=True))

    # Map target column
    if 'Dataset' in df.columns:
        df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})

    return df


def load_kidney_data():
    """Load and preprocess chronic kidney disease dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'kidney_disease.csv'))

    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})

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

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.fillna(df.median(numeric_only=True))
    return df


def load_pancreatic_data():
    """Load and preprocess pancreatic cancer dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'pancreatic_cancer.csv'))

    drop_cols = ['Country', 'Stage_at_Diagnosis', 'Survival_Time_Months', 'Treatment_Type']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}
    for col in ['Physical_Activity_Level', 'Diet_Processed_Food', 'Access_to_Healthcare']:
        if col in df.columns:
            df[col] = df[col].map(ordinal_map)

    if 'Economic_Status' in df.columns:
        df['Economic_Status'] = df['Economic_Status'].map({'Low': 0, 'Middle': 1, 'High': 2})

    if 'Urban_vs_Rural' in df.columns:
        df['Urban_vs_Rural'] = df['Urban_vs_Rural'].map({'Urban': 1, 'Rural': 0})

    return df


def train_model_fallback(disease_type):
    """Train model if pre-trained model not available."""
    print(f"Training {disease_type} model...")

    # Load appropriate data
    if disease_type == 'diabetes':
        df = load_diabetes_data()
        target_col = 'Outcome'
    elif disease_type == 'heart':
        df = load_heart_data()
        target_col = 'Heart Disease'
    elif disease_type == 'parkinsons':
        df = load_parkinsons_data()
        target_col = 'status'
    elif disease_type == 'liver':
        df = load_liver_data()
        target_col = 'Dataset'
    elif disease_type == 'kidney':
        df = load_kidney_data()
        target_col = 'classification'
    elif disease_type == 'pancreatic':
        df = load_pancreatic_data()
        target_col = 'Survival_Status'
    else:
        raise ValueError(f"Unknown disease type: {disease_type}")

    # Prepare features
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    features = list(X.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models and select best
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    # Add SVM for Parkinson's
    if disease_type == 'parkinsons':
        models['SVM'] = SVC(kernel='rbf', probability=True, random_state=42)

    best_model = None
    best_score = 0
    best_name = ''
    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        mean_score = cv_scores.mean()
        results[name] = {
            'cv_mean': mean_score,
            'cv_std': cv_scores.std(),
            'test_score': model.score(X_test_scaled, y_test)
        }

        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, get_model_path(disease_type, 'model'))
    joblib.dump(scaler, get_model_path(disease_type, 'scaler'))
    joblib.dump(features, get_model_path(disease_type, 'features'))
    joblib.dump({'model_name': best_name, 'results': results}, get_model_path(disease_type, 'info'))

    _model_cache[disease_type] = {
        'model': best_model,
        'scaler': scaler,
        'features': features,
        'info': {'model_name': best_name, 'results': results}
    }

    return _model_cache[disease_type]


def make_prediction(disease_type, features_dict):
    """Make prediction using loaded model."""
    model_data = load_model(disease_type)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['features']

    # Create feature array in correct order
    feature_array = np.array([[features_dict.get(f, 0) for f in feature_names]])

    # Scale features
    scaled_features = scaler.transform(feature_array)

    # Make prediction
    prediction = model.predict(scaled_features)[0]

    # Get probability
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(scaled_features)[0]
        probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
    else:
        probability = float(prediction)

    return int(prediction), float(probability)


def get_risk_level(prediction, probability):
    """Determine risk level based on prediction and probability."""
    if prediction == 0:
        return "Low"
    else:
        if probability >= 0.8:
            return "High"
        elif probability >= 0.6:
            return "Medium"
        else:
            return "Medium"


# ============== API ENDPOINTS ==============

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'HealthPredict AI API',
        'version': '1.0.0'
    })


@app.route('/api/diabetes/predict', methods=['POST'])
def predict_diabetes():
    """
    Diabetes Risk Assessment API

    Expected JSON body:
    {
        "Pregnancies": 1,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 25.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 30
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        # Make prediction
        prediction, probability = make_prediction('diabetes', data)
        risk_level = get_risk_level(prediction, probability)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'message': 'High risk of diabetes detected. Please consult a healthcare professional.'
                      if prediction == 1
                      else 'Low risk of diabetes. Maintain a healthy lifestyle.',
            'disease': 'Diabetes'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/heart/predict', methods=['POST'])
def predict_heart():
    """
    Heart Disease Risk Assessment API

    Expected JSON body:
    {
        "Age": 55,
        "Sex": 1,
        "Chest pain type": 2,
        "BP": 140,
        "Cholesterol": 250,
        "FBS over 120": 1,
        "EKG results": 0,
        "Max HR": 150,
        "Exercise angina": 0,
        "ST depression": 1.5,
        "Slope of ST": 2,
        "Number of vessels fluro": 1,
        "Thallium": 3
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
                          'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
                          'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium']

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        # Make prediction
        prediction, probability = make_prediction('heart', data)
        risk_level = get_risk_level(prediction, probability)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'message': 'Indicators suggest potential heart disease risk. Please seek medical evaluation.'
                      if prediction == 1
                      else 'No immediate heart disease risk detected. Continue healthy habits.',
            'disease': 'Heart Disease'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/parkinsons/predict', methods=['POST'])
def predict_parkinsons():
    """
    Parkinson's Disease Screening API

    Expected JSON body with 22 voice biomarker parameters.
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
            'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
            'spread1', 'spread2', 'D2', 'PPE'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        # Make prediction
        prediction, probability = make_prediction('parkinsons', data)
        risk_level = get_risk_level(prediction, probability)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'message': "Voice patterns suggest potential Parkinson's indicators. Consult a neurologist."
                      if prediction == 1
                      else "Voice patterns appear normal. No Parkinson's indicators detected.",
            'disease': "Parkinson's Disease"
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/liver/predict', methods=['POST'])
def predict_liver():
    """
    Liver Health Analysis API

    Expected JSON body:
    {
        "Age": 45,
        "Gender": 1,
        "Total_Bilirubin": 1.5,
        "Direct_Bilirubin": 0.5,
        "Alkaline_Phosphotase": 200,
        "Alamine_Aminotransferase": 30,
        "Aspartate_Aminotransferase": 35,
        "Total_Protiens": 7.0,
        "Albumin": 4.0,
        "Albumin_and_Globulin_Ratio": 1.2
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                          'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                          'Aspartate_Aminotransferase', 'Total_Protiens',
                          'Albumin', 'Albumin_and_Globulin_Ratio']

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        # Make prediction
        prediction, probability = make_prediction('liver', data)
        risk_level = get_risk_level(prediction, probability)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'message': 'Liver function indicators suggest potential concern. Please consult a hepatologist.'
                      if prediction == 1
                      else 'Liver function indicators appear normal. Maintain a healthy lifestyle.',
            'disease': 'Liver Disease'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/model/info/<disease_type>', methods=['GET'])
def get_model_info(disease_type):
    """Get information about a specific model."""
    valid_types = ['diabetes', 'heart', 'parkinsons', 'liver', 'kidney', 'pancreatic']

    if disease_type not in valid_types:
        return jsonify({
            'success': False,
            'error': f'Invalid disease type. Must be one of: {valid_types}'
        }), 400

    try:
        model_data = load_model(disease_type)
        info = model_data['info']

        return jsonify({
            'success': True,
            'disease_type': disease_type,
            'model_name': info.get('model_name', 'Unknown'),
            'features': model_data['features'],
            'feature_count': len(model_data['features'])
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============== KIDNEY DISEASE ENDPOINT ==============

@app.route('/api/kidney/predict', methods=['POST'])
def predict_kidney():
    """
    Chronic Kidney Disease Assessment API

    Expected JSON body:
    {
        "age": 50, "bp": 80, "sg": 1.020, "al": 0, "su": 0,
        "rbc": 1, "pc": 1, "pcc": 0, "ba": 0, "bgr": 120,
        "bu": 40, "sc": 1.2, "sod": 140, "pot": 4.5, "hemo": 13,
        "pcv": 40, "wc": 8000, "rc": 5.0, "htn": 0, "dm": 0,
        "cad": 0, "appet": 1, "pe": 0, "ane": 0
    }
    """
    try:
        data = request.get_json()

        required_fields = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
                          'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
                          'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        prediction, probability = make_prediction('kidney', data)
        risk_level = get_risk_level(prediction, probability)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'message': 'Indicators suggest potential chronic kidney disease. Please consult a nephrologist.'
                      if prediction == 1
                      else 'Kidney function indicators appear normal. Maintain a healthy lifestyle.',
            'disease': 'Chronic Kidney Disease'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============== PANCREATIC CANCER ENDPOINT ==============

@app.route('/api/pancreatic/predict', methods=['POST'])
def predict_pancreatic():
    """
    Pancreatic Cancer Risk Screening API

    Expected JSON body:
    {
        "Age": 55, "Gender": 1, "Smoking_History": 0, "Obesity": 0,
        "Diabetes": 0, "Chronic_Pancreatitis": 0, "Family_History": 0,
        "Hereditary_Condition": 0, "Jaundice": 0, "Abdominal_Discomfort": 0,
        "Back_Pain": 0, "Weight_Loss": 0, "Development_of_Type2_Diabetes": 0,
        "Alcohol_Consumption": 0, "Physical_Activity_Level": 1,
        "Diet_Processed_Food": 1, "Access_to_Healthcare": 2,
        "Urban_vs_Rural": 1, "Economic_Status": 1
    }
    """
    try:
        data = request.get_json()

        required_fields = ['Age', 'Gender', 'Smoking_History', 'Obesity', 'Diabetes',
                          'Chronic_Pancreatitis', 'Family_History', 'Hereditary_Condition',
                          'Jaundice', 'Abdominal_Discomfort', 'Back_Pain', 'Weight_Loss',
                          'Development_of_Type2_Diabetes', 'Alcohol_Consumption',
                          'Physical_Activity_Level', 'Diet_Processed_Food',
                          'Access_to_Healthcare', 'Urban_vs_Rural', 'Economic_Status']

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        prediction, probability = make_prediction('pancreatic', data)
        risk_level = get_risk_level(prediction, probability)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'message': 'Risk factors suggest elevated pancreatic cancer risk. Please consult an oncologist.'
                      if prediction == 1
                      else 'Low pancreatic cancer risk based on current indicators. Maintain a healthy lifestyle.',
            'disease': 'Pancreatic Cancer'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Pre-load all models on startup
    print("Loading models...")
    for disease in ['diabetes', 'heart', 'parkinsons', 'liver', 'kidney', 'pancreatic']:
        try:
            load_model(disease)
            print(f"  - {disease} model loaded successfully")
        except Exception as e:
            print(f"  - {disease} model: {e}")

    port = int(os.environ.get('PORT', 5050))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    print(f"\nStarting Flask API server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)
