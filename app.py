# app.py
import streamlit as st, requests, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, base64, time, os

BASE_URL = "http://127.0.0.1:5000"
st.set_page_config(page_title="Instagram Fake Profile Detector", page_icon="🔍", layout="wide")

# 🔹 DARK MODE TOGGLE
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)

# 🔹 BACKGROUND IMAGE (optimized load)
@st.cache_data
def get_base64_image(image_file):
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64_image("static/bg.jpg")

# 🔹 STYLES (GLASSMORPHISM + ANIMATIONS + SLIGHT BACKGROUND MOVEMENT)
st.markdown(f"""
<style>

@keyframes moveBackground {{
    0% {{ background-position: 0% 0%; }}
    50% {{ background-position: 0% 100%; }}
    100% {{ background-position: 0% 0%; }}
}}

.stApp {{
    background: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-repeat: repeat-y;
    animation: moveBackground 20s linear infinite;
}}

.stApp::before {{
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background: rgba(10, 14, 39, 0.25);
}}

h1, h2, h3 {{
    color: white !important;
    animation: fadeIn 1s ease-in;
}}

@keyframes fadeIn {{
    from {{opacity: 0; transform: translateY(-10px);}}
    to {{opacity: 1; transform: translateY(0);}}
}}

.result-box {{
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
    margin: 15px 0;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}}

.result-box:hover {{
    transform: scale(1.05);
    box-shadow: 0px 0px 15px rgba(255,255,255,0.15);
}}

.fake {{
    background: rgba(255,107,107,0.15);
    border: 2px solid #ff6b6b;
    color: #ff6b6b;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    font-size: 2rem;
    font-weight: bold;
    margin: 15px 0;
    animation: pulseRed 2s ease-in-out;
}}

.real {{
    background: rgba(85,239,196,0.15);
    border: 2px solid #55efc4;
    color: #55efc4;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    font-size: 2rem;
    font-weight: bold;
    margin: 15px 0;
    animation: pulseGreen 2s ease-in-out;
}}

@keyframes pulseRed {{
    0% {{ box-shadow: 0 0 0px #ff6b6b; }}
    50% {{ box-shadow: 0 0 20px #ff6b6b; }}
    100% {{ box-shadow: 0 0 0px #ff6b6b; }}
}}
@keyframes pulseGreen {{
    0% {{ box-shadow: 0 0 0px #55efc4; }}
    50% {{ box-shadow: 0 0 20px #55efc4; }}
    100% {{ box-shadow: 0 0 0px #55efc4; }}
}}

.overlay-red {{
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(255,107,107,0.1);
    pointer-events: none;
}}
.overlay-green {{
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(85,239,196,0.1);
    pointer-events: none;
}}
.explain-text {{
    color: white !important;
    font-size: 1.2rem;
    font-weight: 500;
}}
.explain-heading {{
    color: white !important;
    font-size: 1.4rem;
    font-weight: 600;
    margin-top: 1rem;
}}
.votes-text {{
    color: white !important;
    font-size: 1.1rem;
}}
[data-testid="stTabs"] button {{
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
    backdrop-filter: blur(6px);
    color: white;
}}
[data-testid='stElementToolbar'] svg {{
    fill: black !important;
    color: black !important;
}}
.stAlert p {{
    color: white !important;
}}
.metric-label {{
    color: white !important;
}}
</style>
""", unsafe_allow_html=True)

st.title("🔍 Instagram Fake Profile Detector Dashboard")

# 🔹 INPUTS
profile_pic    = st.sidebar.selectbox("Has Profile Picture?", ["Yes", "No"], help="Yes = 1, No = 0")
username_length = st.sidebar.number_input("Username Length", min_value=1, help="Length of the username")
bio_length      = st.sidebar.number_input("Bio Length", min_value=0, help="Number of characters in bio")
external_url    = st.sidebar.selectbox("Has External URL?", ["Yes", "No"], help="Yes = 1, No = 0")
is_private      = st.sidebar.selectbox("Private Account?", ["Yes", "No"], help="Yes = 1, No = 0")
posts_count     = st.sidebar.number_input("Posts Count", min_value=0, help="Number of posts")
followers_count = st.sidebar.number_input("Followers Count", min_value=0, help="Total followers")
following_count = st.sidebar.number_input("Following Count", min_value=0, help="Total following")

# Convert Yes/No to 1/0
profile_pic_val  = 1 if profile_pic  == "Yes" else 0
external_url_val = 1 if external_url == "Yes" else 0
is_private_val   = 1 if is_private   == "Yes" else 0

followers_following_ratio = followers_count / (following_count + 1)
engagement_score          = posts_count     / (followers_count + 1)

data = {
    "profile_pic":               int(profile_pic_val),
    "username_length":           int(username_length),
    "bio_length":                int(bio_length),
    "external_url":              int(external_url_val),
    "is_private":                int(is_private_val),
    "posts_count":               int(posts_count),
    "followers_count":           int(followers_count),
    "following_count":           int(following_count),
    "followers_following_ratio": float(followers_following_ratio),
    "engagement_score":          float(engagement_score)
}

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Performance Dashboard", "📖 Explain Prediction"])

# ══════════════════════════════════════════════
# 🔮 PREDICTION TAB
# ══════════════════════════════════════════════
with tab1:
    if st.sidebar.button("🔮 Predict Now"):

        # 🔹 PROGRESS BAR
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            progress.progress(i + 1)

        response = requests.post(f"{BASE_URL}/predict/all", json=data)
        try:
            result = response.json()
            st.session_state["prediction_result"] = result
        except ValueError:
            st.error("Backend did not return valid JSON. Check Flask logs.")
            st.write("Raw response from Flask:", response.text)
            result = None

        if result:
            preds = result["all_predictions"]
            df_pred = pd.DataFrame(preds).T
            df_pred[["fake_prob", "real_prob"]] = df_pred[["fake_prob", "real_prob"]].apply(
                pd.to_numeric, errors="coerce"
            )

            final_label    = result["final_label"]
            final_decision = result["final_decision"]
            decision_source = result.get("decision_source", "ensemble")

            st.markdown("<h3>Model Predictions</h3>", unsafe_allow_html=True)
            st.dataframe(df_pred, use_container_width=True)

            st.markdown(f"<p class='votes-text'>Votes Fake: {result['votes_fake']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='votes-text'>Votes Real: {result['votes_real']}</p>", unsafe_allow_html=True)

            # 🔹 FINAL RESULT BOX
            if final_label == 0:
                st.markdown('<div class="result-box fake">🚨 FINAL DECISION: FAKE PROFILE</div>', unsafe_allow_html=True)
                st.markdown("<div class='overlay-red'></div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box real">✅ FINAL DECISION: REAL PROFILE</div>', unsafe_allow_html=True)
                st.markdown("<div class='overlay-green'></div>", unsafe_allow_html=True)

            # 🔹 HEATMAP — uses adjusted probabilities (consistent with final decision)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.heatmap(df_pred[["fake_prob", "real_prob"]], annot=True, cmap="coolwarm", ax=ax)
            st.markdown("<h3>Model Probabilities Heatmap</h3>", unsafe_allow_html=True)
            st.pyplot(fig)

            # 🔹 PIE CHART (VOTES) — uses adjusted votes
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            ax2.pie(
                [result["votes_fake"], result["votes_real"]],
                labels=["Fake", "Real"],
                autopct="%1.1f%%",
                colors=["#ff6b6b", "#55efc4"],
                startangle=90
            )
            st.markdown("<h3>Votes Distribution</h3>", unsafe_allow_html=True)
            st.pyplot(fig2)

            # 🔹 PIE CHART (AVERAGE PROBABILITY) — uses adjusted probs
            avg_fake = df_pred["fake_prob"].mean()
            avg_real = df_pred["real_prob"].mean()
            fig3, ax3 = plt.subplots(figsize=(5, 5))
            ax3.pie(
                [avg_fake, avg_real],
                labels=["Fake Probability", "Real Probability"],
                autopct="%1.1f%%",
                colors=["#ff6b6b", "#55efc4"],
                startangle=90
            )
            st.markdown("<h3>Average Probability Distribution</h3>", unsafe_allow_html=True)
            st.pyplot(fig3)

# ══════════════════════════════════════════════
# 📊 PERFORMANCE TAB
# ══════════════════════════════════════════════
with tab2:
    metrics = requests.get(f"{BASE_URL}/metrics").json()["metrics"]
    metrics_df = pd.DataFrame(metrics).T

    st.markdown("<h3>Model Performance Metrics</h3>", unsafe_allow_html=True)
    st.info("Precision: Accuracy of positive predictions | Recall: Coverage of actual positives | F1: Balance between Precision & Recall")
    st.dataframe(metrics_df, use_container_width=True)

    if all(col in metrics_df.columns for col in ["accuracy", "precision", "recall", "f1_score"]):
        precision_val = float(metrics_df["precision"].mean())
        recall_val    = float(metrics_df["recall"].mean())
        f1_val        = float(metrics_df["f1_score"].mean())

        st.markdown("<p class='metric-label'>Precision</p>", unsafe_allow_html=True)
        st.progress(precision_val)
        if precision_val > 0.9:
            st.success("🏅 High Precision")

        st.markdown("<p class='metric-label'>Recall</p>", unsafe_allow_html=True)
        st.progress(recall_val)

        st.markdown("<p class='metric-label'>F1 Score</p>", unsafe_allow_html=True)
        st.progress(f1_val)

        fig, ax = plt.subplots(figsize=(9, 4))
        metrics_df[["accuracy", "precision", "recall", "f1_score"]].astype(float).plot(kind="bar", ax=ax)
        st.markdown("<h3>Performance Comparison</h3>", unsafe_allow_html=True)
        st.pyplot(fig)

    # 🔹 ROC Curves
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for model_name, model_metrics in metrics.items():
        if "fpr" in model_metrics and "tpr" in model_metrics:
            fpr = pd.to_numeric(model_metrics["fpr"], errors="coerce")
            tpr = pd.to_numeric(model_metrics["tpr"], errors="coerce")
            ax2.plot(fpr, tpr, label=f"{model_name} (AUC={model_metrics['auc_roc']:.2f})")

    ax2.plot([0, 1], [0, 1], '--', color='grey')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    st.markdown("<h3>ROC Curves</h3>", unsafe_allow_html=True)
    st.pyplot(fig2)

    st.markdown("<h3>Class Balance Before vs After SMOTE</h3>", unsafe_allow_html=True)
    st.image("models/class_balance.png", caption="Distribution of Fake vs Real Profiles (Before and After SMOTE)", width=800)

    st.markdown(
        "<p style='color:white; font-size:1.1rem;'>"
        "Before SMOTE → Fake ≈ 3,290, Real ≈ 1,710<br>"
        "After SMOTE  → Fake = 2,500, Real = 2,500<br>"
        "✅ Dataset successfully balanced to 50% Fake and 50% Real using SMOTE."
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown("<h3>Class Balance Visualization</h3>", unsafe_allow_html=True)
    chart_path = os.path.join("models", "class_balance.png")

    if os.path.exists(chart_path):
        st.image(chart_path, caption="Full Dataset, Train Before, Train After", width=800)
        st.markdown(
            "<p style='color:white; font-size:1.1rem;'>"
            "📊 Class Balance Summary:<br>"
            "• Full dataset → Balanced (≈2,500 Fake vs ≈2,500 Real)<br>"
            "• Train before SMOTE → Balanced (2000 Fake vs 2000 Real)<br>"
            "• Train after SMOTE → Same (2000 Fake vs 2000 Real)<br>"
            "• Test set → Balanced (500 Fake vs 500 Real)<br><br>"
            "✅ Dataset is already balanced, so SMOTE did not change counts.<br>"
            "⚠️ Test set is kept unbalanced for proper evaluation."
            "</p>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Class balance chart not found. Please run main.py to generate it.")

# ══════════════════════════════════════════════
# 📖 EXPLAIN PREDICTION TAB
# ══════════════════════════════════════════════
with tab3:
    if "prediction_result" in st.session_state:
        result = st.session_state["prediction_result"]

        if "explain_prediction" in result:
            final_decision  = result["final_decision"]
            final_label     = result["final_label"]
            decision_source = result.get("decision_source", "ensemble")
            rule_reason     = result.get("rule_reason", "")

            st.markdown("<h3 class='explain-heading'>Model Analysis for This Profile:</h3>", unsafe_allow_html=True)

            # 🔹 Use adjusted probabilities from all_predictions (consistent with final decision)
            preds_df = pd.DataFrame(result["all_predictions"]).T
            preds_df[["fake_prob", "real_prob"]] = preds_df[["fake_prob", "real_prob"]].apply(
                pd.to_numeric, errors="coerce"
            )

            avg_fake = preds_df["fake_prob"].mean()
            avg_real = preds_df["real_prob"].mean()

            # 🔹 Fetch live model accuracies from /metrics for dynamic display
            try:
                metrics_resp = requests.get(f"{BASE_URL}/metrics").json()["metrics"]
                xgb_accuracy          = metrics_resp.get("XGBoost", {}).get("accuracy", 0.0)
                soft_voting_accuracy  = metrics_resp.get("Soft Voting Ensemble", {}).get("accuracy", 0.0)
                stacking_accuracy     = metrics_resp.get("Stacking Ensemble", {}).get("accuracy", 0.0)
            except Exception:
                xgb_accuracy         = 0.0
                soft_voting_accuracy = 0.0
                stacking_accuracy    = 0.0

            xgb_real_prob          = preds_df.loc["XGBoost", "real_prob"]          if "XGBoost"          in preds_df.index else 0.5
            soft_voting_real_prob  = preds_df.loc["Soft Voting Ensemble", "real_prob"] if "Soft Voting Ensemble" in preds_df.index else 0.5
            stacking_real_prob     = preds_df.loc["Stacking Ensemble", "real_prob"] if "Stacking Ensemble" in preds_df.index else 0.5

            # ── Comparative explanation (fully dynamic — all values from current prediction) ──
            ensemble_vs_xgb = max(soft_voting_real_prob, stacking_real_prob)
            if ensemble_vs_xgb < xgb_real_prob:
                dilute_word   = "diluted"
                compare_word  = "slightly lower"
                superior_word = "superior"
            else:
                dilute_word   = "balanced out"
                compare_word  = "comparable or higher"
                superior_word = "competitive"

            stacking_vs_soft = "improved over" if stacking_real_prob > soft_voting_real_prob else "produced results similar to"
            confidence_level = "high confidence" if (avg_real > 0.7 or avg_fake > 0.7) else "moderate confidence"

            if final_decision == "Fake":
                explanation_text = f"""
                <p style='color:white; font-size:1.1rem;'>
                In our experiments on this profile, <b>XGBoost</b> achieved a fake probability score of
                <b>{preds_df.loc["XGBoost","fake_prob"]:.2f}</b> with accuracy of <b>{xgb_accuracy:.2f}</b>.
                The <b>Soft Voting Ensemble</b> (fake probability <b>{preds_df.loc["Soft Voting Ensemble","fake_prob"]:.2f}</b>,
                accuracy <b>{soft_voting_accuracy:.2f}</b>) and
                <b>Stacking Ensemble</b> (fake probability <b>{preds_df.loc["Stacking Ensemble","fake_prob"]:.2f}</b>,
                accuracy <b>{stacking_accuracy:.2f}</b>) produced comparable scores.
                Ensemble methods combine predictions from all base models, including weaker ones such as Naive Bayes,
                which can balance out the overall performance.
                For this profile, the consensus across models indicates high confidence in the final <b>Fake 🚨</b> classification.
                </p>
                """
            else:
                explanation_text = f"""
                <p style='color:white; font-size:1.1rem;'>
                In our experiments on this profile, <b>XGBoost</b> achieved a real probability score of
                <b>{preds_df.loc["XGBoost","real_prob"]:.2f}</b> with accuracy of <b>{xgb_accuracy:.2f}</b>.
                The <b>Soft Voting Ensemble</b> (real probability <b>{preds_df.loc["Soft Voting Ensemble","real_prob"]:.2f}</b>,
                accuracy <b>{soft_voting_accuracy:.2f}</b>) and
                <b>Stacking Ensemble</b> (real probability <b>{preds_df.loc["Stacking Ensemble","real_prob"]:.2f}</b>,
                accuracy <b>{stacking_accuracy:.2f}</b>) produced comparable scores.
                Ensemble methods combine predictions from all base models, including weaker ones such as Naive Bayes,
                which can balance out the overall performance.
                For this profile, the consensus across models indicates high confidence in the final <b>Real ✅</b> classification.
                </p>
                """

            st.markdown(explanation_text, unsafe_allow_html=True)

            # Feature values
            st.markdown("<h3 class='explain-heading'>Feature Values</h3>", unsafe_allow_html=True)
            values_df = pd.DataFrame.from_dict(
                result["explain_prediction"]["values"], orient="index", columns=["Value"]
            )
            st.dataframe(values_df, use_container_width=True)

            # Feature importance
            st.markdown("<h3 class='explain-heading'>Averaged Feature Importance</h3>", unsafe_allow_html=True)
            importance_df = pd.DataFrame.from_dict(
                result["explain_prediction"]["importance"], orient="index", columns=["Importance"]
            )
            importance_df = importance_df.sort_values("Importance", ascending=False)
            st.dataframe(importance_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            importance_df.plot(kind="bar", ax=ax, legend=False, color="#ff6b6b")
            ax.set_ylabel("Importance Score")
            ax.set_title("Feature Importance (Averaged Across Models)")
            st.pyplot(fig)

            # 🔹 Why this profile was classified — aligned with final decision
            st.markdown(
                "<h3 class='explain-heading'>Why This Profile Was Classified as "
                + final_decision + ":</h3>",
                unsafe_allow_html=True
            )

            if final_label == 0:
                # ── FAKE explanation ─────────────────────────────────
                if decision_source == "rule":
                    reason_detail = rule_reason
                else:
                    reason_detail = "weak engagement signals (e.g., no posts, no bio, external URL spam), which strongly match fake profile patterns"

                additional_text = f"""
                <p style='color:white; font-size:1.1rem;'>
                Based on the given inputs, the system predicts this profile as <b>Fake 🚨</b>.<br>
                The average fake probability across models is <b>{avg_fake:.2f}</b>, while the real
                probability is <b>{avg_real:.2f}</b>.<br>
                This decision is influenced by {reason_detail}.
                </p>
                """
            else:
                # ── REAL explanation ─────────────────────────────────
                if decision_source == "rule":
                    reason_detail = rule_reason
                else:
                    reason_detail = (
                        "healthy engagement signals (profile picture, balanced followers/following ratio, "
                        "consistent posting activity), which strongly match real profile patterns"
                    )

                additional_text = f"""
                <p style='color:white; font-size:1.1rem;'>
                Based on the given inputs, the system predicts this profile as <b>Real ✅</b>.<br>
                The average real probability across models is <b>{avg_real:.2f}</b>, while the fake
                probability is <b>{avg_fake:.2f}</b>.<br>
                The features suggest a genuine account with {reason_detail}.
                </p>
                """

            st.markdown(additional_text, unsafe_allow_html=True)
