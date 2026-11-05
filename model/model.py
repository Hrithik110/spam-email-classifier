from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "model" / "data" / "spambase.data"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

word_freq = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet",
    "order", "mail", "receive", "will", "people", "report", "addresses",
    "free", "business", "email", "you", "credit", "your", "font", "000",
    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
    "data", "415", "85", "technology", "1999", "parts", "pm", "direct",
    "cs", "meeting", "original", "project", "re", "edu", "table", "conference",
]
char_freq = [";", "(", "[", "!", "$", "#"]
columns = (
    [f"word_freq_{w}" for w in word_freq]
    + [f"char_freq_{c}" for c in char_freq]
    + ["capital_run_length_average",
       "capital_run_length_longest",
       "capital_run_length_total",
       "is_spam"]
)

df = pd.read_csv(DATA_PATH, header=None, names=columns)
print(df.shape)
print(df["is_spam"].value_counts())

# ---------- Split ----------
X = df.drop(columns="is_spam")
y = df["is_spam"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Unscaled test set + target, for the Streamlit app
test_df = X_test.copy()
test_df["is_spam"] = y_test
test_df.to_csv(BASE_DIR / "test_data.csv", index=False)

# ---------- Scale (fit on train only) ----------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

# ---------- Models ----------
models = {
    "Logistic Regression": ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42)),
    "Decision Tree": ("decision_tree", DecisionTreeClassifier(random_state=42)),
    "KNN": ("knn", KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": ("naive_bayes", GaussianNB()),
    "Support Vector Machine": ("svm", SVC(probability=True, random_state=42)),
}

rows = []
for name, (file_stem, clf) in models.items():
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]   # probability of spam (class 1)

    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Train Accuracy": clf.score(X_train_s, y_train),  # overfitting check
    })

    joblib.dump(clf, MODEL_DIR / f"{file_stem}.joblib")

results = pd.DataFrame(rows).set_index("ML Model Name")
results.round(4).to_csv(MODEL_DIR / "results.csv")
print(results.round(4))

# ---------- Choosing the best model ----------
best = results["MCC"].idxmax()
print(f"\nBest model by MCC: {best}")
print(results.loc[best].round(4))

# CONCLUSION (fill in after you see YOUR numbers; do not copy blindly):
# Rank by MCC, with F1 and AUC as tie-breakers, because accuracy alone can
# mislead on a 61/39 class split. In spam filtering a false positive (real
# mail marked spam) is costly, so also check Precision and the confusion
# matrix. Compare "Train Accuracy" with test Accuracy to spot overfitting
# (Decision Tree and KNN are the usual suspects). If two models differ by a
# fraction of a percent, call it a tie and prefer the
# simpler or faster one.