# app.py
import streamlit as st, requests, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, base64

BASE_URL = "http://127.0.0.1:5000"
st.set_page_config(page_title="Instagram Fake Profile Detector", page_icon="🔍", layout="wide")

def get_base64_image(image_file):
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()
bg_image = get_base64_image("static/bg.png")

st.markdown(f"""<style>
.stApp {{background: url("data:image/png;base64,{bg_image}");background-size: cover;}}
h1,h2,h3,h4,h5,h6 {{color:#ffffff !important;}}
[data-testid="stTab"] {{color:#ffffff !important;}}
.result-box {{border-radius:12px;padding:20px;text-align:center;font-size:1.4rem;font-weight:bold;margin:10px 0;}}
.fake {{background:rgba(255,107,107,0.15);border:2px solid #ff6b6b;color:#ff6b6b;}}
.real {{background:rgba(85,239,196,0.15);border:2px solid #55efc4;color:#55efc4;}}
</style>""", unsafe_allow_html=True)

st.title("🔍 Instagram Fake Profile Detector Dashboard")

# Sidebar inputs (no defaults, user must enter)
profile_pic = st.sidebar.selectbox("Has Profile Picture?", [1, 0])
username_length = st.sidebar.number_input("Username Length", min_value=1)
bio_length = st.sidebar.number_input("Bio Length", min_value=0)
external_url = st.sidebar.selectbox("Has External URL?", [1, 0])
is_private = st.sidebar.selectbox("Private Account?", [1, 0])
posts_count = st.sidebar.number_input("Posts Count", min_value=0)
followers_count = st.sidebar.number_input("Followers Count", min_value=0)
following_count = st.sidebar.number_input("Following Count", min_value=0)

followers_following_ratio = followers_count / (following_count + 1) if following_count >= 0 else 0
engagement_score = posts_count / (followers_count + 1) if followers_count >= 0 else 0

data = {"profile_pic": int(profile_pic),"username_length": int(username_length),"bio_length": int(bio_length),
        "external_url": int(external_url),"is_private": int(is_private),"posts_count": int(posts_count),
        "followers_count": int(followers_count),"following_count": int(following_count),
        "followers_following_ratio": float(followers_following_ratio),"engagement_score": float(engagement_score)}

tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Performance Dashboard"])

with tab1:
    if st.sidebar.button("🔮 Predict Now"):
        url = f"{BASE_URL}/predict/all"
        response = requests.post(url, json=data)
        result = response.json()

        preds = result["all_predictions"]
        df = pd.DataFrame(preds).T
        df[["fake_prob","real_prob"]] = df[["fake_prob","real_prob"]].apply(pd.to_numeric, errors="coerce")

        st.markdown("<h3 style='color:white;'>Model Predictions</h3>", unsafe_allow_html=True)
        st.dataframe(df)

        st.markdown(f"<p style='color:white;'>Votes Fake: {result['votes_fake']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:white;'>Votes Real: {result['votes_real']}</p>", unsafe_allow_html=True)

        if result["final_label"] == 0:
            st.markdown('<div class="result-box fake">🚨 FINAL DECISION: FAKE PROFILE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box real">✅ FINAL DECISION: REAL PROFILE</div>', unsafe_allow_html=True)

        # Heatmap
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(df[["fake_prob","real_prob"]], annot=True, cmap="coolwarm", ax=ax)
        st.markdown("<h3 style='color:white;'>Model Probabilities Heatmap</h3>", unsafe_allow_html=True)
        st.pyplot(fig)

        # Pie chart of votes
        fig2, ax2 = plt.subplots()
        ax2.pie([result["votes_fake"], result["votes_real"]],
                labels=["Fake","Real"], autopct="%1.1f%%",
                colors=["#ff6b6b","#55efc4"], startangle=90)
        st.markdown("<h3 style='color:white;'>Votes Distribution</h3>", unsafe_allow_html=True)
        st.pyplot(fig2)

        # Pie chart of average probabilities across models
        avg_fake = df["fake_prob"].mean()
        avg_real = df["real_prob"].mean()
        fig3, ax3 = plt.subplots()
        ax3.pie([avg_fake, avg_real],
                labels=["Fake Probability","Real Probability"], autopct="%1.1f%%",
                colors=["#ff6b6b","#55efc4"], startangle=90)
        st.markdown("<h3 style='color:white;'>Average Probability Distribution (All Models)</h3>", unsafe_allow_html=True)
        st.pyplot(fig3)

with tab2:
    metrics = requests.get(f"{BASE_URL}/metrics").json()["metrics"]
    metrics_df = pd.DataFrame(metrics).T
    st.markdown("<h3 style='color:white;'>Model Performance Metrics</h3>", unsafe_allow_html=True)
    st.dataframe(metrics_df)

    if all(col in metrics_df.columns for col in ["accuracy","precision","recall","f1_score"]):
        fig, ax = plt.subplots(figsize=(10,5))
        metrics_df[["accuracy","precision","recall","f1_score"]].astype(float).plot(kind="bar", ax=ax)
        st.markdown("<h3 style='color:white;'>Performance Comparison Bar Chart</h3>", unsafe_allow_html=True)
        st.pyplot(fig)
    else:
        st.markdown("<p style='color:white;'>⚠ Metrics not available for plotting.</p>", unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(8,6))
    for model_name, model_metrics in metrics.items():
        if "fpr" in model_metrics and "tpr" in model_metrics:
            fpr = model_metrics["fpr"]
            tpr = model_metrics["tpr"]
            auc_score = model_metrics["auc_roc"]
            ax2.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.2f})")
    ax2.plot([0,1],[0,1],'--',color='grey')
    st.markdown("<h3 style='color:white;'>ROC Curves - All Models</h3>", unsafe_allow_html=True)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    st.pyplot(fig2)
