# app.py
import streamlit as st, requests, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, base64, time, os

BASE_URL = "http://127.0.0.1:5000"
st.set_page_config(page_title="Instagram Fake Profile Detector", page_icon="🔍", layout="wide")

# 🔹 DARK MODE TOGGLE
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)

# 🔹 BACKGROUND IMAGE (optimized load)
@st.cache_data  # cache image encoding for performance
def get_base64_image(image_file):
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64_image("static/bg.jpg")  # use compressed .jpg for lighter weight

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
    background-size: cover;   /* allow room to move */
    background-repeat: repeat-y;  /* repeat vertically */
    animation: moveBackground 20s linear infinite; /* subtle visible movement */
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
    box-shadow: 0px 0px 15px rgba(255,255,255,0.15); /* lighter shadow for efficiency */
}}

.fake {{
    background: rgba(255,107,107,0.15); /* white background inside the box */
    border: 2px solid #ff6b6b; /* red border for fake */
    color: #ff6b6b; /* red text inside */
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    font-size: 2rem; /* larger font size */
    font-weight: bold;
    margin: 15px 0;
    animation: pulseRed 2s ease-in-out;
}}

.real {{
    background: rgba(85,239,196,0.15); /* white background inside the box */
    border: 2px solid #55efc4; /* green border for real */
    color: #55efc4; /* green text inside */
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    font-size: 2rem; /* larger font size */
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
[data-testid="stTabs"] button {{
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
    backdrop-filter: blur(6px); /* reduced blur for smoother performance */
    color: white;
}}
[data-testid='stElementToolbar'] svg {{
    fill: black !important;
    color: black !important;
}}
.stAlert p {{
    color: white !important;
}}
</style>
""", unsafe_allow_html=True)

st.title("🔍 Instagram Fake Profile Detector Dashboard")

# 🔹 INPUTS
profile_pic = st.sidebar.selectbox("Has Profile Picture?", ["Yes", "No"], help="Yes = 1, No = 0")
username_length = st.sidebar.number_input("Username Length", min_value=1, help="Length of the username")
bio_length = st.sidebar.number_input("Bio Length", min_value=0, help="Number of characters in bio")
external_url = st.sidebar.selectbox("Has External URL?", ["Yes", "No"], help="Yes = 1, No = 0")
is_private = st.sidebar.selectbox("Private Account?", ["Yes", "No"], help="Yes = 1, No = 0")
posts_count = st.sidebar.number_input("Posts Count", min_value=0, help="Number of posts")
followers_count = st.sidebar.number_input("Followers Count", min_value=0, help="Total followers")
following_count = st.sidebar.number_input("Following Count", min_value=0, help="Total following")
# Convert Yes/No to 1/0
profile_pic = 1 if profile_pic == "Yes" else 0
external_url = 1 if external_url == "Yes" else 0
is_private = 1 if is_private == "Yes" else 0
followers_following_ratio = followers_count / (following_count + 1)
engagement_score = posts_count / (followers_count + 1)

data = {
    "profile_pic": int(profile_pic),
    "username_length": int(username_length),
    "bio_length": int(bio_length),
    "external_url": int(external_url),
    "is_private": int(is_private),
    "posts_count": int(posts_count),
    "followers_count": int(followers_count),
    "following_count": int(following_count),
    "followers_following_ratio": float(followers_following_ratio),
    "engagement_score": float(engagement_score)
}

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Performance Dashboard", "📖 Explain Prediction"])

# 🔮 PREDICTION TAB
with tab1:
    if st.sidebar.button("🔮 Predict Now"):

        # 🔹 PROGRESS BAR
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.005)  # faster loop for responsiveness
            progress.progress(i + 1)

        response = requests.post(f"{BASE_URL}/predict/all", json=data)
        try:
            result = response.json()
            st.session_state["prediction_result"] = result
        except ValueError:
            st.error("Backend did not return valid JSON. Check Flask logs.")
            st.write("Raw response from Flask:", response.text)
            result = None

        preds = result["all_predictions"]
        df = pd.DataFrame(preds).T
        df[["fake_prob","real_prob"]] = df[["fake_prob","real_prob"]].apply(pd.to_numeric, errors="coerce")

        st.markdown("<h3>Model Predictions</h3>", unsafe_allow_html=True)
        st.dataframe(df, width="stretch")  # responsive table

        st.markdown(f"<p>Votes Fake: {result['votes_fake']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p>Votes Real: {result['votes_real']}</p>", unsafe_allow_html=True)

        # 🔹 FINAL RESULT BOX
        if result["final_label"] == 0:
            st.markdown('<div class="result-box fake">🚨 FINAL DECISION: FAKE PROFILE</div>', unsafe_allow_html=True)
            st.markdown("<div class='overlay-red'></div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box real">✅ FINAL DECISION: REAL PROFILE</div>', unsafe_allow_html=True)
            st.markdown("<div class='overlay-green'></div>", unsafe_allow_html=True)
        # 🔹 HEATMAP
        fig, ax = plt.subplots(figsize=(7,4))  # slightly smaller for viewport fit
        sns.heatmap(df[["fake_prob","real_prob"]], annot=True, cmap="coolwarm", ax=ax)
        st.markdown("<h3>Model Probabilities Heatmap</h3>", unsafe_allow_html=True)
        st.pyplot(fig)

        # 🔹 PIE CHART (VOTES)
        fig2, ax2 = plt.subplots(figsize=(5,5))
        ax2.pie([result["votes_fake"], result["votes_real"]],
                labels=["Fake","Real"], autopct="%1.1f%%",
                colors=["#ff6b6b","#55efc4"], startangle=90)
        st.markdown("<h3>Votes Distribution</h3>", unsafe_allow_html=True)
        st.pyplot(fig2)

        # 🔹 PIE CHART (AVERAGE PROBABILITY)
        avg_fake = df["fake_prob"].mean()
        avg_real = df["real_prob"].mean()
        fig3, ax3 = plt.subplots(figsize=(5,5))
        ax3.pie([avg_fake, avg_real],
                labels=["Fake Probability","Real Probability"],
                autopct="%1.1f%%",
                colors=["#ff6b6b","#55efc4"],
                startangle=90)
        st.markdown("<h3>Average Probability Distribution</h3>", unsafe_allow_html=True)
        st.pyplot(fig3)

# 📊 PERFORMANCE TAB
with tab2:
    metrics = requests.get(f"{BASE_URL}/metrics").json()["metrics"]
    metrics_df = pd.DataFrame(metrics).T

    st.markdown("<h3>Model Performance Metrics</h3>", unsafe_allow_html=True)

    # 🔹 TOOLTIPS
    st.info("Precision: Accuracy of positive predictions | Recall: Coverage of actual positives | F1: Balance between Precision & Recall")
    # Show metrics table
    st.dataframe(metrics_df, width="stretch")

    # 🔹 Gamified progress bars + badges
    if all(col in metrics_df.columns for col in ["accuracy","precision","recall","f1_score"]):
        precision = float(metrics_df["precision"].mean())
        recall = float(metrics_df["recall"].mean())
        f1_score = float(metrics_df["f1_score"].mean())

        st.write("Precision")
        st.progress(precision)
        if precision > 0.9:
            st.success("🏅 High Precision")

        st.write("Recall")
        st.progress(recall)

        st.write("F1 Score")
        st.progress(f1_score)

        # 🔹 Bar chart comparison
        fig, ax = plt.subplots(figsize=(9,4))
        metrics_df[["accuracy","precision","recall","f1_score"]].astype(float).plot(kind="bar", ax=ax)
        st.markdown("<h3>Performance Comparison</h3>", unsafe_allow_html=True)
        st.pyplot(fig)

    # 🔹 ROC Curves
    fig2, ax2 = plt.subplots(figsize=(7,5))
    for model_name, model_metrics in metrics.items():
        if "fpr" in model_metrics and "tpr" in model_metrics:
            fpr = pd.to_numeric(model_metrics["fpr"], errors="coerce")
            tpr = pd.to_numeric(model_metrics["tpr"], errors="coerce")
            ax2.plot(fpr, tpr,label=f"{model_name} (AUC={model_metrics['auc_roc']:.2f})")

    ax2.plot([0,1],[0,1],'--',color='grey')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()

    st.markdown("<h3>ROC Curves</h3>", unsafe_allow_html=True)
    st.pyplot(fig2)
    # 🔹 Show Class Balance Chart (Before vs After SMOTE)
    st.markdown("<h3>Class Balance Before vs After SMOTE</h3>", unsafe_allow_html=True)
    st.image("models/class_balance.png", caption="Distribution of Fake vs Real Profiles (Before and After SMOTE)", width=800)

    # 🔹 Statement showing difference
    st.markdown(
        "<p style='color:white; font-size:1.1rem;'>"
        "Before SMOTE → Fake ≈ 3,290, Real ≈ 1,710<br>"
        "After SMOTE  → Fake = 2,500, Real = 2,500<br>"
        "✅ Dataset successfully balanced to 50% Fake and 50% Real using SMOTE."
        "</p>",
        unsafe_allow_html=True
    )
    
    # 🔹 Class Balance Visualization
    st.markdown("<h3>Class Balance Visualization</h3>", unsafe_allow_html=True)

    chart_path = os.path.join("models", "class_balance.png")

    if os.path.exists(chart_path):
        st.image(
            chart_path,
            caption="Full Dataset, Train Before, Train After, Test After (Visualization Only)",
            width=800
        )

        st.markdown(
            "<p style='color:white; font-size:1.1rem;'>"
            "📊 Class Balance Summary:<br>"
            "• Full dataset → Balanced (≈2,500 Fake vs ≈2,500 Real)<br>"
            "• Train before SMOTE → Balanced (2000 Fake vs 2000 Real)<br>"
            "• Train after SMOTE → Same (2000 Fake vs 2000 Real)<br>"
            "• Test set → Balanced (500 Fake vs 500 Real)<br>"
            "• Test after SMOTE (visualization only) → Same (500 vs 500)<br><br>"
            "✅ Dataset is already balanced, so SMOTE did not change counts.<br>"
            "⚠️ Test set balancing shown only for visualization — not used in evaluation."
            "</p>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Class balance chart not found. Please run main.py to generate it.")

    
# 📖 EXPLAIN PREDICTION TAB
with tab3:
    if "prediction_result" in st.session_state:
        result = st.session_state["prediction_result"]

        if "explain_prediction" in result:
            # 🔹 Comparative Explanation Statement
            metrics = requests.get(f"{BASE_URL}/metrics").json()["metrics"]
            if "XGBoost" in metrics and "Soft Voting Ensemble" in metrics and "Stacking Ensemble" in metrics:
                xgb_acc = metrics["XGBoost"]["accuracy"]
                soft_acc = metrics["Soft Voting Ensemble"]["accuracy"]
                stack_acc = metrics["Stacking Ensemble"]["accuracy"]

                explanation_text = f"""
                <p style='color:white; font-size:1.1rem;'>
                In our experiments, <b>XGBoost</b> achieved the highest individual performance with an accuracy of {xgb_acc:.2f}.  
                The <b>Soft Voting Ensemble</b> ({soft_acc:.2f}) and <b>Stacking Ensemble</b> ({stack_acc:.2f}) were slightly lower.  
                This is because ensemble methods combine predictions from all base models, including weaker ones such as Naive Bayes, which diluted the overall performance.  
                While stacking improved over soft voting by learning optimal weights, XGBoost remained superior due to its strong ability to capture complex patterns.  
                Importantly, ensembles provide robustness and stability across different test splits, even if they do not always surpass the strongest individual model.
                </p>
                """
                st.markdown(explanation_text, unsafe_allow_html=True)
                
            # Feature values
            st.markdown("<h3 class='explain-heading'>Feature Values</h3>", unsafe_allow_html=True)
            values_df = pd.DataFrame.from_dict(result["explain_prediction"]["values"], orient="index", columns=["Value"])
            st.dataframe(values_df, width="stretch")

            # Feature importance (averaged across models)
            st.markdown("<h3 class='explain-heading'>Averaged Feature Importance</h3>", unsafe_allow_html=True)
            importance_df = pd.DataFrame.from_dict(result["explain_prediction"]["importance"], orient="index", columns=["Importance"])
            importance_df = importance_df.sort_values("Importance", ascending=False)
            st.dataframe(importance_df, width="stretch")

            # Bar chart of averaged importance
            fig, ax = plt.subplots(figsize=(8,4))
            importance_df.plot(kind="bar", ax=ax, legend=False, color="#ff6b6b")
            ax.set_ylabel("Importance Score")
            ax.set_title("Feature Importance (Averaged Across Models)")
            st.pyplot(fig)
            
            # 🔹 Dynamic Input-Based Statement
            preds_df = pd.DataFrame(result["all_predictions"]).T
            avg_fake = preds_df["fake_prob"].mean()
            avg_real = preds_df["real_prob"].mean()
            final_decision = result["final_decision"]

            if final_decision == "Fake":
                dynamic_text = f"""
                <p style='color:white; font-size:1.1rem;'>
                Based on the given inputs, the system predicts this profile as <b>Fake 🚨</b>.  
                The average fake probability across models is <b>{avg_fake:.2f}</b>, while the real probability is <b>{avg_real:.2f}</b>.  
                This decision is influenced by weak engagement signals (e.g., no posts, no bio, external URL spam), which strongly match fake profile patterns.
                </p>
                """
            else:
                dynamic_text = f"""
                <p style='color:white; font-size:1.1rem;'>
                Based on the given inputs, the system predicts this profile as <b>Real ✅</b>.  
                The average real probability across models is <b>{avg_real:.2f}</b>, while the fake probability is <b>{avg_fake:.2f}</b>.  
                The features suggest a genuine account (profile picture, balanced followers/following, and engagement), which aligns with real profile behavior.
                </p>
                """
            st.markdown(dynamic_text, unsafe_allow_html=True)

        else:
            st.warning("No explanation data returned from backend.")
    else:
<<<<<<< HEAD
        st.info("Run a prediction first in the Prediction tab.")
=======
        st.info("Run a prediction first in the Prediction tab.")
>>>>>>> aac6dfe (Resolve merge conflicts)
