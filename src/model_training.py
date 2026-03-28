import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os

def train_and_save_model(X_train, y_train, X_test, y_test, scaler, feature_names):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc'
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("=== Résultats du modèle ===")
    print(f"Accuracy     : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC      : {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Sauvegarde
    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }, 'models/best_model.pkl')
    
    print("✅ Modèle sauvegardé avec succès !")
    return model