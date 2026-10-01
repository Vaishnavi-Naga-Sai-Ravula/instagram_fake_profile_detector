# app.py
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64

BASE_URL = "http://127.0.0.1:5000"

st.set_page_config(page_title="Instagram Fake Profile Detector", page_icon="🔍", layout="wide")

# ✅ Background setup
def get_base64_image(image_file):
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64_image("static/bg.png")

st.markdown(f"""
<style>
.stApp {{
    background: url("data:image/png;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    position: relative;
}}
.stApp::before {{
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background: rgba(10, 14, 39, 0.85);
    backdrop-filter: blur(3px);
    z-index: 0;
}}
.stApp > * {{
    position: relative;
    z-index: 1;
}}
h1 {{ color: #ffffff !important; }}
h3 {{ color: #ffffff !important; }}
div.block-container p {{ color: #ffffff !important; }}
[data-testid="stMetricLabel"] {{ color: #ffffff !important; }}
[data-testid="stMetricValue"] {{ color: #ffffff !important; }}
.result-box {{
    border-radius: 12px; padding: 20px; text-align: center;
    font-size: 1.4rem; font-weight: bold; margin: 10px 0;
}}
.fake {{ background: rgba(255,107,107,0.15); border: 2px solid #ff6b6b; color: #ff6b6b; }}
.real {{ background: rgba(85,239,196,0.15); border: 2px solid #55efc4; color: #55efc4; }}
</style>
""", unsafe_allow_html=True)

st.title("🔍 Instagram Fake Profile Detector Dashboard")
st.write("Frontend powered by Streamlit · Backend powered by Flask API")

# Sidebar inputs
profile_pic = st.sidebar.selectbox("Has Profile Picture?", [1, 0])
username_length = st.sidebar.number_input("Username Length", min_value=1, value=10)
bio_length = st.sidebar.number_input("Bio Length", min_value=0, value=50)
external_url = st.sidebar.selectbox("Has External URL?", [1, 0])
is_private = st.sidebar.selectbox("Private Account?", [1, 0])
posts_count = st.sidebar.number_input("Posts Count", min_value=0, value=50)
followers_count = st.sidebar.number_input("Followers Count", min_value=0, value=1000)
following_count = st.sidebar.number_input("Following Count", min_value=0, value=500)

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

tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Performance Dashboard"])

with tab1:
    if st.sidebar.button("🔮 Predict Now"):
        url = f"{BASE_URL}/predict/all"
        response = requests.post(url, json=data)
        result = response.json()

        preds = result["all_predictions"]
        df = pd.DataFrame(preds).T
        df[["fake_prob","real_prob"]] = df[["fake_prob","real_prob"]].apply(pd.to_numeric, errors="coerce")

        # ✅ White heading
        st.markdown("<h3 style='color:white;'>Model Predictions</h3>", unsafe_allow_html=True)
        st.dataframe(df)

        # ✅ Text format for votes
        st.markdown(f"<p style='color:white;'>Votes Fake: {result['votes_fake']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:white;'>Votes Real: {result['votes_real']}</p>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(df[["fake_prob","real_prob"]], annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Model Probabilities Heatmap")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.pie([result["votes_fake"], result["votes_real"]],
                labels=["Fake","Real"], autopct="%1.1f%%",
                colors=["#ff6b6b","#55efc4"], startangle=90)
        ax2.set_title("Votes Distribution")
        st.pyplot(fig2)

with tab2:
    metrics = requests.get(f"{BASE_URL}/metrics").json()["metrics"]
    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df)

    fig, ax = plt.subplots(figsize=(10,5))
    metrics_df[["accuracy","precision","recall","f1_score"]].astype(float).plot(kind="bar", ax=ax)
    ax.set_title("Model Performance Metrics")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(8,6))
    for model_name, model_metrics in metrics.items():
        if "fpr" in model_metrics and "tpr" in model_metrics:
            fpr = model_metrics["fpr"]
            tpr = model_metrics["tpr"]
            auc_score = model_metrics["auc_roc"]
            ax2.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.2f})")
    ax2.plot([0,1],[0,1],'--',color='grey')
    ax2.set_title("ROC Curves - All Models")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    st.pyplot(fig2)
