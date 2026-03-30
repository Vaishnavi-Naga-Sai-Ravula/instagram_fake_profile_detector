# flask_app.py
from flask import Flask, request, jsonify
import joblib, numpy as np, os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

app = Flask(__name__)

models = {
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl"),
    "Voting Ensemble": joblib.load("models/voting.pkl"),
    "AdaBoost": joblib.load("models/adaboost.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl")
}

scaler = joblib.load("models/scaler.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

X_test = joblib.load("models/X_test.pkl") if os.path.exists("models/X_test.pkl") else None
y_test = joblib.load("models/y_test.pkl") if os.path.exists("models/y_test.pkl") else None

MODEL_METRICS = {}
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

@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({"metrics": MODEL_METRICS})

@app.route("/predict/all", methods=["POST"])
def predict_all():
    data = request.json
    features = np.array([[data[col] for col in feature_cols]]).astype(float)
    features_scaled = scaler.transform(features)

    results, votes_fake, votes_real = {}, 0, 0
    for name, model in models.items():
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0]
        results[name] = {"prediction": int(pred),
                            "fake_prob": float(prob[0]),
                            "real_prob": float(prob[1])}
        if pred == 0: votes_fake += 1
        else: votes_real += 1

    final_label = 0 if votes_fake > votes_real else 1
    final_decision = "Fake" if final_label == 0 else "Real"

    return jsonify({
        "all_predictions": results,
        "votes_fake": votes_fake,
        "votes_real": votes_real,
        "final_label": final_label,
        "final_decision": final_decision
    })

if __name__ == "__main__":
    app.run(debug=True)
