# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score, precision_score, recall_score)

# Models - 4 CORE MODELS ONLY
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from xgboost import XGBClassifier

print("="*70)
print("INSTAGRAM FAKE PROFILE DETECTION - TRAINING PIPELINE")
print("🤖 Using 4 Core Models: NB, RF, XGB, ADA")
print("="*70)

if not os.path.exists('models'):
    os.makedirs('models')

# ════════════════════════════════════════════════════════════
# LOAD DATASET WITH ROBUST ENCODING
# ════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading Dataset...")

def load_csv_safe(filepath):
    """Load CSV with multiple encoding fallbacks"""
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip', engine='python')
            print(f"✓ Successfully loaded with {encoding} encoding")
            return df
        except Exception as e:
            continue
    
    raise ValueError("Could not load CSV file with any encoding")

df = load_csv_safe("data/instagram_fake_profile.csv")
print(f"Dataset Shape: {df.shape}")
print(f"Available Columns: {df.columns.tolist()}\n")

# ════════════════════════════════════════════════════════════
# NORMALIZE COLUMN NAMES
# ════════════════════════════════════════════════════════════
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print(f"Normalized Columns: {df.columns.tolist()}\n")

# ════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════
print("[STEP 2] Feature Engineering...")

# Ensure required columns exist
required_cols = ['followers_count', 'following_count', 'posts_count']
for col in required_cols:
    if col not in df.columns:
        print(f"⚠ Missing column: {col}")
        df[col] = 0

# Create derived features
df['followers_following_ratio'] = df['followers_count'] / (df['following_count'] + 1)
df['engagement_score'] = df['posts_count'] / (df['followers_count'] + 1)

# Handle bio description length
if 'bio_description_length' in df.columns:
    df['bio_length'] = df['bio_description_length']
else:
    df['bio_length'] = 0

print("Feature engineering complete ✓\n")

# ════════════════════════════════════════════════════════════
# PREPROCESSING
# ═════════════════════════════════════════════════════════���══
print("[STEP 3] Preprocessing...")

df.fillna(0, inplace=True)

# Encode label (handle multiple possible column names)
label_col = None
for possible_name in ['fake_or_real', 'label_fake_or_real', 'label']:
    if possible_name in df.columns:
        label_col = possible_name
        break

if label_col is None:
    print("⚠ Label column not found!")
    print(f"Available columns: {df.columns.tolist()}")
    raise ValueError("Label column not found")

le = LabelEncoder()
df['label'] = le.fit_transform(df[label_col].astype(str))
print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

# ════════════════════════════════════════════════════════════
# SELECT FEATURES
# ════════════════════════════════════════════════════════════
print("[STEP 4] Selecting Features...")

FEATURE_COLS = [
    'profile_pic',
    'username_length',
    'bio_length',
    'external_url',
    'is_private',
    'posts_count',
    'followers_count',
    'following_count',
    'followers_following_ratio',
    'engagement_score',
]

# Ensure all feature columns exist
for col in FEATURE_COLS:
    if col not in df.columns:
        print(f"⚠ Creating missing column: {col}")
        df[col] = 0

X = df[FEATURE_COLS].fillna(0)
y = df['label']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Class distribution:\n{y.value_counts()}\n")

# ════════════════════════════════════════════════════════════
# TRAIN-TEST SPLIT
# ════════════════════════════════════════════════════════════
print("[STEP 5] Train-Test Split...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}\n")

# ════════════════════════════════════════════════════════════
# FEATURE SCALING
# ════════════════════════════════════════════════════════════
print("[STEP 6] Feature Scaling...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling complete ✓\n")

# ════════════════════════════════════════════════════════════
# TRAIN 4 CORE MODELS
# ════════════════════════════════════════════════════════════
print("[STEP 7] Training 4 Core Models...")

models = {
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=150, eval_metric='logloss', random_state=42, verbosity=0),
    "AdaBoost": AdaBoostClassifier(n_estimators=150, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {"Accuracy": round(acc, 4), "F1 Score": round(f1, 4)}
    print(f"  ✓ {name:<24} Acc: {acc:.4f} | F1: {f1:.4f}")

# ════════════════════════════════════════════════════════════
# VOTING ENSEMBLE (4 Models)
# ════════════════════════════════════════════════════════════
print("\n[STEP 8] Training Voting Ensemble...")

ensemble = VotingClassifier(
    estimators=[
        ('nb', GaussianNB()),
        ('rf', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=150, eval_metric='logloss', random_state=42, verbosity=0)),
        ('ada', AdaBoostClassifier(n_estimators=150, random_state=42)),
    ],
    voting='soft',
    weights=[1, 2, 2, 1]
).fit(X_train_scaled, y_train)

y_pred_ensemble = ensemble.predict(X_test_scaled)
acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
f1_ensemble = f1_score(y_test, y_pred_ensemble)
results["Voting Ensemble"] = {"Accuracy": round(acc_ensemble, 4), "F1 Score": round(f1_ensemble, 4)}

print(f"  ✓ Voting Ensemble       Acc: {acc_ensemble:.4f} | F1: {f1_ensemble:.4f}")

# ════════════════════════════════════════════════════════════
# RESULTS TABLE
# ════════════════════════════════════════════════════════════
print("\n[STEP 9] Model Comparison Results")
print("="*70)

results_df = pd.DataFrame(results).T
print(results_df)
print("="*70 + "\n")

# ════════════════════════════════════════════════════════════
# DETAILED METRICS
# ════════════════════════════════════════════════════════════
print("[STEP 10] Detailed Metrics for All Models")
print("="*70)

all_models_dict = {**models, "Voting Ensemble": ensemble}
for name, model in all_models_dict.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    
    print(f"\n{name}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  AUC-ROC:   {auc_score:.4f}")

print("="*70 + "\n")

# ════════════════════════════════════════════════════════════
# SAVE MODELS
# ════════════════════════════════════════════════════════════
print("\n[STEP 11] Saving Models...")

with open('models/ensemble.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
    print("  ✓ Saved: models/ensemble.pkl")

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    print("  ✓ Saved: models/scaler.pkl")

with open('models/feature_cols.pkl', 'wb') as f:
    pickle.dump(FEATURE_COLS, f)
    print("  ✓ Saved: models/feature_cols.pkl")

# Save individual models
with open('models/naive_bayes.pkl', 'wb') as f:
    pickle.dump(models["Naive Bayes"], f)
    print("  ✓ Saved: models/naive_bayes.pkl")

with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(models["Random Forest"], f)
    print("  ✓ Saved: models/random_forest.pkl")

with open('models/xgboost.pkl', 'wb') as f:
    pickle.dump(models["XGBoost"], f)
    print("  ✓ Saved: models/xgboost.pkl")

with open('models/adaboost.pkl', 'wb') as f:
    pickle.dump(models["AdaBoost"], f)
    print("  ✓ Saved: models/adaboost.pkl")

print("\n" + "="*70)
print("✅ TRAINING PIPELINE COMPLETE!")
print("="*70)
print("\n📊 Summary:")
print(f"  • Dataset: {len(df):,} samples")
print(f"  • Features: {len(FEATURE_COLS)}")
print(f"  • Models: 4 Core Models + Voting Ensemble")
print(f"  • Best Model: {results_df['Accuracy'].idxmax()} ({results_df['Accuracy'].max():.4f})")
print(f"  • Training Complete: ✅")
print("="*70)