import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("🔮 Prédiction de Churn Client - Telco")

# Charger le modèle
@st.cache_resource
def load_model():
    try:
        data = joblib.load("models/best_model.pkl")
        encoders = joblib.load("models/label_encoders.pkl")
        return data['model'], data['scaler'], data['feature_names'], encoders
    except:
        st.error("Modèle non trouvé. Veuillez d'abord exécuter le notebook d'entraînement.")
        st.stop()

model, scaler, feature_names, encoders = load_model()

st.success("✅ Modèle chargé avec succès")

# Formulaire
st.sidebar.header("📋 Informations du Client")

tenure = st.sidebar.slider("Ancienneté (mois)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Montant mensuel ($)", 18.0, 118.0, 70.0)
total_charges = st.sidebar.slider("Montant total ($)", 0.0, 9000.0, 800.0)

contract = st.sidebar.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Méthode de paiement", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
internet_service = st.sidebar.selectbox("Service Internet", ["DSL", "Fiber optic", "No"])

# Nouvelles options ajoutées (très importantes pour le modèle)
online_security = st.sidebar.selectbox("Sécurité en ligne", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Support technique", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Sauvegarde en ligne", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Protection des appareils", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# Autres variables fixes (tu peux les rendre interactives plus tard si tu veux)
gender = st.sidebar.selectbox("Genre", ["Male", "Female"], index=0)
senior = st.sidebar.checkbox("Senior Citizen", value=False)
partner = st.sidebar.selectbox("A un partenaire ?", ["Yes", "No"], index=1)
dependents = st.sidebar.selectbox("A des personnes à charge ?", ["Yes", "No"], index=1)

if st.button("🚀 Prédire le Churn", type="primary"):
    
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': 1 if senior else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': 'Yes',
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    # Encodage
    input_encoded = input_data.copy()
    for col, encoder in encoders.items():
        if col in input_encoded.columns:
            try:
                input_encoded[col] = encoder.transform(input_encoded[col])
            except:
                input_encoded[col] = 0

    # Réordonner exactement comme dans l'entraînement
    input_encoded = input_encoded[feature_names]

    # Scaling + Prédiction
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Affichage
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error(f"⚠️ Client à **risque de churn**")
        else:
            st.success(f"✅ Client **fidèle**")
    
    with col2:
        st.metric("Probabilité de Churn", f"{probability:.1%}")