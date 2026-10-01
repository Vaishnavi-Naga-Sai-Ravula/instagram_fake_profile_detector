# train_models.py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Example dataset (replace with your Instagram profile dataset)
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================== Random Forest ==================
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
joblib.dump(rf, "models/random_forest.pkl")

# ================== XGBoost ==================
from xgboost import XGBClassifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(X_train, y_train)
joblib.dump(xgb, "models/xgboost.pkl")

# ================== AdaBoost ==================
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
joblib.dump(ada, "models/adaboost.pkl")

# ================== Naive Bayes ==================
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
joblib.dump(nb, "models/naive_bayes.pkl")

# ================== Voting Ensemble ==================
from sklearn.ensemble import VotingClassifier

estimators = [
    ("rf", rf),
    ("xgb", xgb),
    ("ada", ada),
    ("nb", nb)
]

voting = VotingClassifier(estimators=estimators, voting="soft")
voting.fit(X_train, y_train)
joblib.dump(voting, "models/voting.pkl")

# ================== Save test data ==================
joblib.dump(X_test, "models/X_test.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print("✅ All models and test data saved successfully!")
