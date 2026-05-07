# flask_app.py
from flask import Flask, request, jsonify
import joblib, numpy as np, os, json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

app = Flask(__name__)

# 🔹 Load trained models
models = {
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl"),
    "Soft Voting Ensemble": joblib.load("models/soft_voting.pkl"),
    "Stacking Ensemble": joblib.load("models/stacking.pkl"),
    "AdaBoost": joblib.load("models/adaboost.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl")
}

# 🔹 Load scaler and feature columns
scaler = joblib.load("models/scaler.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

# 🔹 Load test data (optional fallback if metrics.json not found)
X_test = joblib.load("models/X_test.pkl") if os.path.exists("models/X_test.pkl") else None
y_test = joblib.load("models/y_test.pkl") if os.path.exists("models/y_test.pkl") else None

# 🔹 Load precomputed metrics from JSON
MODEL_METRICS = {}
if os.path.exists("models/metrics.json"):
    with open("models/metrics.json", "r") as f:
        MODEL_METRICS = json.load(f)
else:
    # Fallback: compute metrics dynamically if JSON not found
    if X_test is not None and y_test is not None:
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]
            MODEL_METRICS[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "auc_roc": roc_auc_score(y_test, y_prob),
                "fpr": roc_curve(y_test, y_prob)[0].tolist(),
                "tpr": roc_curve(y_test, y_prob)[1].tolist()
            }

# 🔹 Metrics endpoint
@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({"metrics": MODEL_METRICS})

# 🔹 Prediction endpoint
@app.route("/predict/all", methods=["POST"])
def predict_all():
    data = request.json

    # Derived features
    data["followers_following_ratio"] = data["followers_count"] / (data["following_count"] + 1)
    data["engagement_score"] = data["posts_count"] / (data["followers_count"] + 1)
    data["posts_per_follower"] = data["posts_count"] / (data["followers_count"] + 1)
    data["followers_per_post"] = data["followers_count"] / (data["posts_count"] + 1)

    # Build feature array
    features = np.array([[data[col] for col in feature_cols]]).astype(float)
    features_scaled = scaler.transform(features)

    results, votes_fake, votes_real = {}, 0, 0
    for name, model in models.items():
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0]
        results[name] = {
            "prediction": int(pred),
            "fake_prob": float(prob[0]),
            "real_prob": float(prob[1])
        }
        if pred == 0:
            votes_fake += 1
        else:
            votes_real += 1
    # 🔹 Ensemble decision
    final_label = 0 if votes_fake > votes_real else 1
    final_decision = "Fake" if final_label == 0 else "Real"
    # 🔹 Simple rule layer for obvious fakes
    # Case 1: Spam bots (no pic, no bio, mass following)
    if data["profile_pic"] == 0 and data["bio_length"] == 0 and data["followers_count"] < 50 and data["following_count"] > 1000:
        final_label = 0
        final_decision = "Fake"

    # Case 2: Empty accounts (no posts, external URL spam)
    elif data["posts_count"] == 0 and data["external_url"] == 1:
        final_label = 0
        final_decision = "Fake"

    # Case 3: Extreme imbalance (too many followers but no engagement)
    elif data["followers_count"] > 10000 and data["posts_count"] < 5:
        final_label = 0
        final_decision = "Fake"
        
    # 🔹 Explain prediction (feature values)
    explain_values = {col: float(val) for col, val in zip(feature_cols, features[0])}

    # 🔹 Feature importance across multiple models
    importance_list = []

    # Random Forest
    if hasattr(models["Random Forest"], "feature_importances_"):
        importance_list.append(models["Random Forest"].feature_importances_)

    # XGBoost
    if hasattr(models["XGBoost"], "feature_importances_"):
        importance_list.append(models["XGBoost"].feature_importances_)

    # AdaBoost
    if hasattr(models["AdaBoost"], "feature_importances_"):
        importance_list.append(models["AdaBoost"].feature_importances_)
        
    # Naive Bayes (approx importance using log probabilities)
    if hasattr(models["Naive Bayes"], "feature_log_prob_"):
        nb_importance = np.abs(models["Naive Bayes"].feature_log_prob_).mean(axis=0)
        importance_list.append(nb_importance)
        
    # Average importance if available
    if importance_list:
        avg_importance = np.mean(importance_list, axis=0)
        explain_importance = {col: float(score) for col, score in zip(feature_cols, avg_importance)}
    else:
        explain_importance = {}

    return jsonify({
        "all_predictions": results,
        "votes_fake": votes_fake,
        "votes_real": votes_real,
        "final_label": final_label,
        "final_decision": final_decision,
        "explain_prediction": {
            "values": explain_values,
            "importance": explain_importance
        }
    })

if __name__ == "__main__":
    app.run(debug=True)