# train.py  (à créer à la racine du projet)
from src.data_preprocessing import load_and_preprocess
from src.model_training import train_and_save_model

print("🚀 Lancement de l'entraînement du modèle...")

# Chargement et prétraitement des données
X_train, X_test, y_train, y_test, scaler, feature_names, encoders = load_and_preprocess()

print(f"✅ Données prêtes → X_train shape: {X_train.shape}")

# Entraînement et sauvegarde du modèle
model = train_and_save_model(X_train, y_train, X_test, y_test, scaler, feature_names)

print("🎉 Modèle entraîné et sauvegardé avec succès dans le dossier models/ !")
print("Tu peux maintenant lancer Streamlit.")