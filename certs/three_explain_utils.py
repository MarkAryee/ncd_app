


# ----------------------------
# Centralized SHAP + Interpretation
# ----------------------------
import shap
import numpy as np
import pandas as pd

def explain_prediction(model, scaler, input_df, X, disease_name="NCD"):
    """
    Centralized function to handle:
    - Scaling input
    - SHAP explainability
    - Risk interpretation
    - Final verdict
    """
    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict probability and class
    proba = float(model.predict(input_scaled).flatten()[0])
    pred = int(model.predict(input_scaled)[0])

    # -------------------------------------------------------------------------
    # SHAP Explainability
    # ----------------------------
    background = scaler.transform(input_df.sample(1, random_state=42).values)  # small ref sample
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(input_scaled)

    # shap_values is [class0, class1] for binary
    shap_for_user = shap_values[0][0]
    
   
    feature_names = list(X.columns)
    
    # Contributions
    feature_contribs = sorted(
        zip(feature_names, shap_for_user),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # ----------------------------
    # Interpretation
    # ----------------------------
    percent = proba * 100
    if percent < 10:
        category = "Low"
        verdict = (
            f"Your predicted risk of {disease_name} is low. "
            f"This means your features look similar to people in the dataset who rarely developed {disease_name.lower()}."
        )
    elif percent < 25:
        category = "Moderate"
        verdict = (
            f"Your predicted risk of {disease_name} is moderate. "
            f"Even though the number (e.g., {percent:.1f}%) looks small, it is higher than average for people of similar profile in the dataset."
        )
    else:
        category = "High"
        verdict = (
            f"Your predicted risk of {disease_name} is high. "
            f"Compared to similar people in the dataset, your profile places you in a group that developed {disease_name.lower()} more often."
        )

    # Why this result (Top 5 features)
    why_reasons = []
    for feat, val in feature_contribs[:5]:
        if val > 0:
            reason = "pushed the risk upward"
        else:
            reason = "helped lower the risk"
        why_reasons.append(f"- {feat}: {val:.4f} → {reason}")

    # Return unified response
    return {
        "ncd_probability": proba,
        "ncd_prediction": pred,
        "category": category,
        "verdict": verdict,
        "top_features": feature_contribs[:5],
        "why_this_result": why_reasons
    }


