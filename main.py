# main.py
import pandas as pd, numpy as np, pickle, os, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

if not os.path.exists('models'):
    os.makedirs('models')

df = pd.read_csv("data/instagram_fake_profile.csv", encoding="utf-8", on_bad_lines="skip")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

df['followers_following_ratio'] = df['followers_count'] / (df['following_count'] + 1)
df['engagement_score'] = df['posts_count'] / (df['followers_count'] + 1)
df['bio_length'] = df.get('bio_description_length', 0)

le = LabelEncoder()
df['label'] = le.fit_transform(df['fake_or_real'].astype(str))

FEATURE_COLS = [
    'profile_pic','username_length','bio_length','external_url','is_private',
    'posts_count','followers_count','following_count',
    'followers_following_ratio','engagement_score'
]

X = df[FEATURE_COLS].fillna(0)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=150, eval_metric='logloss', random_state=42, verbosity=0),
    "AdaBoost": AdaBoostClassifier(n_estimators=150, random_state=42),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

ensemble = VotingClassifier(
    estimators=[('nb', models["Naive Bayes"]),
                ('rf', models["Random Forest"]),
                ('xgb', models["XGBoost"]),
                ('ada', models["AdaBoost"])],
    voting='soft', weights=[1,2,2,1]
).fit(X_train_scaled, y_train)

# Save models and test data
pickle.dump(ensemble, open("models/voting.pkl","wb"))
pickle.dump(scaler, open("models/scaler.pkl","wb"))
pickle.dump(FEATURE_COLS, open("models/feature_cols.pkl","wb"))
pickle.dump(X_test_scaled, open("models/X_test.pkl","wb"))
pickle.dump(y_test, open("models/y_test.pkl","wb"))
for name, model in models.items():
    pickle.dump(model, open(f"models/{name.lower().replace(' ','_')}.pkl","wb"))