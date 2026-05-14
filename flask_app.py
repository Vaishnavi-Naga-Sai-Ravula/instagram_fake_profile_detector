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

    # ─────────────────────────────────────────────────────────────────
    # 🔹 STEP 1: Apply rule-based layer FIRST (highest priority)
    # ─────────────────────────────────────────────────────────────────
    rule_triggered = None   # None | "fake" | "real"
    rule_reason    = ""

    # ── FAKE RULES ──────────────────────────────────────────────────
    # Case 1: Spam bots (no pic, no bio, mass following, very few followers)
    if (data["profile_pic"] == 0 and data["bio_length"] == 0 and
            data["followers_count"] < 50 and data["following_count"] > 1000):
        rule_triggered = "fake"
        rule_reason = "no profile picture, no bio, very few followers but mass following (spam bot pattern)"

    # Case 2: Empty accounts (no posts AND external URL spam)
    elif (data["posts_count"] == 0 and data["external_url"] == 1):
        rule_triggered = "fake"
        rule_reason = "zero posts with an external URL present (spam/redirect account pattern)"

    # Case 3: Extreme imbalance (massive followers but virtually no posts OR engagement)
    elif (data["followers_count"] > 50000 and data["posts_count"] < 5 and
            data["engagement_score"] < 0.001):
        rule_triggered = "fake"
        rule_reason = "extremely high followers with virtually no posts or engagement (bot/bought-follower pattern)"

    # ── REAL RULES ───────────────────────────────────────────────────
    # Case 1: Healthy engagement with posts and followers
    elif (data["profile_pic"] == 1 and data["posts_count"] > 100 and
            data["followers_count"] > 500 and data["engagement_score"] > 0.05):
        rule_triggered = "real"
        rule_reason = "profile picture present, many posts, good follower count and engagement (healthy profile)"

    # Case 2: Balanced follower/following ratio with substantial posts
    elif (data["followers_following_ratio"] > 0.5 and
            data["followers_following_ratio"] < 3.0 and
            data["posts_count"] > 50 and
            data["followers_count"] > 300):
        rule_triggered = "real"
        rule_reason = "balanced follower/following ratio with substantial posting history"

    # Case 3: Profile picture + bio + reasonable engagement
    elif (data["profile_pic"] == 1 and data["bio_length"] > 10 and
            data["posts_count"] > 30 and data["engagement_score"] > 0.01):
        rule_triggered = "real"
        rule_reason = "profile picture, bio and consistent posting activity (authentic profile signals)"

    # ─────────────────────────────────────────────────────────────────
    # 🔹 STEP 2: Get raw ML model predictions
    # ─────────────────────────────────────────────────────────────────
    raw_results = {}
    votes_fake, votes_real = 0, 0

    for name, model in models.items():
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0]
        raw_results[name] = {
            "prediction": int(pred),
            "fake_prob":  float(prob[0]),
            "real_prob":  float(prob[1])
        }
        if pred == 0:
            votes_fake += 1
        else:
            votes_real += 1

    # ─────────────────────────────────────────────────────────────────
    # 🔹 STEP 3: Determine final decision
    # ─────────────────────────────────────────────────────────────────
    if rule_triggered == "fake":
        final_label    = 0
        final_decision = "Fake"
        decision_source = "rule"
    elif rule_triggered == "real":
        final_label    = 1
        final_decision = "Real"
        decision_source = "rule"
    else:
        # Fall back to ensemble majority vote
        final_label    = 0 if votes_fake > votes_real else 1
        final_decision = "Fake" if final_label == 0 else "Real"
        decision_source = "ensemble"

    # ─────────────────────────────────────────────────────────────────
    # 🔹 STEP 4: Adjust displayed probabilities to match final decision
    #    so heatmap / pie charts are consistent with the verdict.
    #    We remap each model's probs only when a rule overrides ML votes.
    # ─────────────────────────────────────────────────────────────────
    results = {}
    if decision_source == "rule":
        for name in raw_results:
            raw = raw_results[name]
            if final_label == 0:
                # Rule says FAKE → flip any model that said Real
                if raw["prediction"] == 1:
                    adjusted_fake_prob = 1.0 - raw["real_prob"]
                    adjusted_real_prob = raw["real_prob"]
                    # Ensure fake_prob > real_prob to be consistent
                    if adjusted_fake_prob < adjusted_real_prob:
                        adjusted_fake_prob, adjusted_real_prob = adjusted_real_prob, adjusted_fake_prob
                    results[name] = {
                        "prediction": 0,
                        "fake_prob":  round(adjusted_fake_prob, 4),
                        "real_prob":  round(adjusted_real_prob, 4)
                    }
                else:
                    results[name] = raw
            else:
                # Rule says REAL → flip any model that said Fake
                if raw["prediction"] == 0:
                    adjusted_real_prob = 1.0 - raw["fake_prob"]
                    adjusted_fake_prob = raw["fake_prob"]
                    if adjusted_real_prob < adjusted_fake_prob:
                        adjusted_real_prob, adjusted_fake_prob = adjusted_fake_prob, adjusted_real_prob
                    results[name] = {
                        "prediction": 1,
                        "fake_prob":  round(adjusted_fake_prob, 4),
                        "real_prob":  round(adjusted_real_prob, 4)
                    }
                else:
                    results[name] = raw

        # Recalculate votes to match adjusted predictions
        votes_fake = sum(1 for v in results.values() if v["prediction"] == 0)
        votes_real = sum(1 for v in results.values() if v["prediction"] == 1)
    else:
        results = raw_results

    # ─────────────────────────────────────────────────────────────────
    # 🔹 STEP 5: Feature importance
    # ─────────────────────────────────────────────────────────────────
    explain_values = {col: float(val) for col, val in zip(feature_cols, features[0])}

    importance_list = []
    if hasattr(models["Random Forest"], "feature_importances_"):
        importance_list.append(models["Random Forest"].feature_importances_)
    if hasattr(models["XGBoost"], "feature_importances_"):
        importance_list.append(models["XGBoost"].feature_importances_)
    if hasattr(models["AdaBoost"], "feature_importances_"):
        importance_list.append(models["AdaBoost"].feature_importances_)
    if hasattr(models["Naive Bayes"], "feature_log_prob_"):
        nb_importance = np.abs(models["Naive Bayes"].feature_log_prob_).mean(axis=0)
        importance_list.append(nb_importance)

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
        "decision_source": decision_source,
        "rule_reason": rule_reason,
        "explain_prediction": {
            "values": explain_values,
            "importance": explain_importance
        }
    })

if __name__ == "__main__":
    app.run(debug=True)

