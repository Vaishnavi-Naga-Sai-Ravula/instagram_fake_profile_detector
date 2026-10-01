# flask_app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

app = Flask(__name__)

# Load models
models = {
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl"),
    "Voting Ensemble": joblib.load("models/voting.pkl"),
    "AdaBoost": joblib.load("models/adaboost.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl")
}

# Load test data
X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")

# Compute metrics
MODEL_METRICS = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    MODEL_METRICS[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc_roc": auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }

@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({"metrics": MODEL_METRICS})

@app.route("/predict/all", methods=["POST"])
def predict_all():
    data = request.json
    features = np.array([list(data.values())]).astype(float)

    results = {}
    votes_fake, votes_real = 0, 0

    for name, model in models.items():
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]
        results[name] = {
            "prediction": int(pred),
            "fake_prob": float(prob[0]),
            "real_prob": float(prob[1])
        }
        if pred == 0:
            votes_fake += 1
        else:
            votes_real += 1

    return jsonify({
        "all_predictions": results,
        "votes_fake": votes_fake,
        "votes_real": votes_real
    })

if __name__ == "__main__":
    app.run(debug=True)
