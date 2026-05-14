# main.py
import pandas as pd, numpy as np, pickle, os, warnings, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

if not os.path.exists('models'):
    os.makedirs('models')

# 🔹 Load and clean dataset
df = pd.read_csv("data/instagram_fake_profile.csv", encoding="utf-8", on_bad_lines="skip")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# 🔹 Feature Engineering
df['followers_following_ratio'] = df['followers_count'] / (df['following_count'] + 1)
df['engagement_score'] = df['posts_count'] / (df['followers_count'] + 1)
df['bio_length'] = df.get('bio_description_length', 0)
df['posts_per_follower'] = df['posts_count'] / (df['followers_count'] + 1)
df['followers_per_post'] = df['followers_count'] / (df['posts_count'] + 1)

# 🔹 Label Encoding
le = LabelEncoder()
df['label'] = le.fit_transform(df['fake_or_real'].astype(str))

FEATURE_COLS = [
    'profile_pic','username_length','bio_length','external_url','is_private',
    'posts_count','followers_count','following_count',
    'followers_following_ratio','engagement_score',
    'posts_per_follower','followers_per_post'
]

X = df[FEATURE_COLS].fillna(0)
y = df['label']

# 🔹 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 🔹 Show distributions before SMOTE
print("Full dataset:", y.value_counts())
print("Train set before SMOTE:", y_train.value_counts())
print("Test set:", y_test.value_counts())

# 🔹 Apply SMOTE to training set (for actual model training)
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print("Train set after SMOTE:", y_train_balanced.value_counts())

# 🔹 Visualize distributions
fig, axes = plt.subplots(1, 3, figsize=(15,4))

# Full dataset
y.value_counts().plot(kind="bar", ax=axes[0], color=["#ff6b6b","#55efc4"])
axes[0].set_title("Full Dataset")
axes[0].set_xticklabels(["Fake (0)", "Real (1)"], rotation=0)

# Train before SMOTE
y_train.value_counts().plot(kind="bar", ax=axes[1], color=["#ff6b6b","#55efc4"])
axes[1].set_title("Train Before SMOTE")
axes[1].set_xticklabels(["Fake (0)", "Real (1)"], rotation=0)

# Train after SMOTE
y_train_balanced.value_counts().plot(kind="bar", ax=axes[2], color=["#ff6b6b","#55efc4"])
axes[2].set_title("Train After SMOTE")
axes[2].set_xticklabels(["Fake (0)", "Real (1)"], rotation=0)

plt.tight_layout()
plt.savefig("models/class_balance.png")
plt.close()

# 🔹 Statement showing difference
print("\n📊 Class Balance Difference:")
print(f"Full dataset → Fake: {y.value_counts()[0]}, Real: {y.value_counts()[1]}")
print(f"Train before SMOTE → Fake: {y_train.value_counts()[0]}, Real: {y_train.value_counts()[1]}")
print(f"Train after SMOTE → Fake: {y_train_balanced.value_counts()[0]}, Real: {y_train_balanced.value_counts()[1]}")
print(f"Test set → Fake: {y_test.value_counts()[0]}, Real: {y_test.value_counts()[1]}")

# 🔹 Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# 🔹 Base Models (tuned)
models = {
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, eval_metric='logloss', random_state=42, verbosity=0),
    "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.8, random_state=42),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train_balanced)

# 🔹 Soft Voting Ensemble
soft_voting = VotingClassifier(
    estimators=[('nb', models["Naive Bayes"]),
                ('rf', models["Random Forest"]),
                ('xgb', models["XGBoost"]),
                ('ada', models["AdaBoost"])],
    voting='soft',
    n_jobs=1
).fit(X_train_scaled, y_train_balanced)

# 🔹 Stacking Ensemble
stacking = StackingClassifier(
    estimators=[('nb', models["Naive Bayes"]),
                ('rf', models["Random Forest"]),
                ('xgb', models["XGBoost"]),
                ('ada', models["AdaBoost"])],
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method='predict_proba',
    passthrough=True,
    n_jobs=1
).fit(X_train_scaled, y_train_balanced)

# 🔹 Evaluate all models + ensemble
MODEL_METRICS = {}
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:,1]
    MODEL_METRICS[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "fpr": roc_curve(y_test, y_prob)[0].tolist(),
        "tpr": roc_curve(y_test, y_prob)[1].tolist()
    }

# Soft Voting metrics
y_pred = soft_voting.predict(X_test_scaled)
y_prob = soft_voting.predict_proba(X_test_scaled)[:,1]
MODEL_METRICS["Soft Voting Ensemble"] = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "auc_roc": roc_auc_score(y_test, y_prob),
    "fpr": roc_curve(y_test, y_prob)[0].tolist(),
    "tpr": roc_curve(y_test, y_prob)[1].tolist()
}

# Stacking metrics
y_pred = stacking.predict(X_test_scaled)
y_prob = stacking.predict_proba(X_test_scaled)[:,1]
MODEL_METRICS["Stacking Ensemble"] = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "auc_roc": roc_auc_score(y_test, y_prob),
    "fpr": roc_curve(y_test, y_prob)[0].tolist(),
    "tpr": roc_curve(y_test, y_prob)[1].tolist()
}

# 🔹 Save metrics to JSON file
with open("models/metrics.json", "w") as f:
    json.dump(MODEL_METRICS, f, indent=4)

# 🔹 Save models and test data
pickle.dump(soft_voting, open("models/soft_voting.pkl","wb"))
pickle.dump(stacking, open("models/stacking.pkl","wb"))
pickle.dump(scaler, open("models/scaler.pkl","wb"))
pickle.dump(FEATURE_COLS, open("models/feature_cols.pkl","wb"))
pickle.dump(X_test_scaled, open("models/X_test.pkl","wb"))
pickle.dump(y_test, open("models/y_test.pkl","wb"))
for name, model in models.items():
    pickle.dump(model, open(f"models/{name.lower().replace(' ','_')}.pkl","wb"))
