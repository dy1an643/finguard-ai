"""
FinGuard AI – Credit Risk & Fraud Detection
============================================
Dataset: German Credit Risk (Kaggle)
  https://www.kaggle.com/datasets/uciml/german-credit

Save as 'german_credit_data.csv' in the same folder, then:
    python finguard_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("  FinGuard AI – Credit Risk & Fraud Detection")
print("=" * 60)


# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("\n[1] Loading dataset...")
try:
    df = pd.read_csv("german_credit_data.csv", index_col=0)
except FileNotFoundError:
    raise SystemExit("  ERROR: 'german_credit_data.csv' not found in current folder.")

df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
print(f"    Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print(f"    Columns: {list(df.columns)}")


# ── 2. Add Risk column if missing ─────────────────────────────────────────────
print("\n[2] Checking for target column...")

known_targets = {"Risk", "Creditability", "credit_risk"}
if not known_targets.intersection(df.columns):
    print("    No target column found — generating 'Risk' from feature rules...")
    score = pd.Series(0.0, index=df.index)
    if "Checking account" in df.columns:
        score += df["Checking account"].map(
            {"little": 2, "moderate": 1, "rich": 0}).fillna(3)
    if "Saving accounts" in df.columns:
        score += df["Saving accounts"].map(
            {"little": 2, "moderate": 1, "quite rich": 0, "rich": 0}).fillna(2)
    if "Duration" in df.columns:
        score += (df["Duration"] / df["Duration"].max()) * 2
    if "Credit amount" in df.columns:
        score += (df["Credit amount"] / df["Credit amount"].max()) * 2
    if "Age" in df.columns:
        score += ((df["Age"].max() - df["Age"]) / df["Age"].max())
    score_norm = (score - score.min()) / (score.max() - score.min())
    df["Risk"] = (score_norm > 0.45).astype(int)
    print(f"    Generated — Good(0): {(df['Risk']==0).sum()}  Bad(1): {(df['Risk']==1).sum()}")

# Normalise column name to 'Risk'
if "Creditability" in df.columns:
    df = df.rename(columns={"Creditability": "Risk"})
elif "credit_risk" in df.columns:
    df = df.rename(columns={"credit_risk": "Risk"})

# Normalise target values to 0/1
if df["Risk"].dtype == object:
    df["Risk"] = df["Risk"].str.strip().str.lower().map({"good": 0, "bad": 1})
else:
    uv = set(df["Risk"].dropna().astype(int).unique())
    if uv == {1, 2}:
        df["Risk"] = df["Risk"].map({1: 0, 2: 1})

df.dropna(subset=["Risk"], inplace=True)
df["Risk"] = df["Risk"].astype(int)
print(f"    Final class counts:\n{df['Risk'].value_counts().to_string()}")


# ── 3. Data Cleaning ──────────────────────────────────────────────────────────
print("\n[3] Cleaning data...")

for col in df.select_dtypes(include="object").columns:
    mode = df[col].mode()
    if not mode.empty:
        df[col].fillna(mode.iloc[0], inplace=True)
for col in df.select_dtypes(include="number").columns:
    df[col].fillna(df[col].median(), inplace=True)

before = len(df)
df.drop_duplicates(inplace=True)
print(f"    Duplicates removed: {before - len(df)}")
print(f"    Shape after cleaning: {df.shape}")


# ── 4. Encode Categorical Columns ────────────────────────────────────────────
print("\n[4] Encoding categorical columns...")

le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
print(f"    Encoded {len(cat_cols)} columns: {cat_cols}")


# ── 5. Feature Engineering ────────────────────────────────────────────────────
print("\n[5] Feature engineering...")

if "Credit amount" in df.columns and "Duration" in df.columns:
    df["Monthly_Payment"] = df["Credit amount"] / df["Duration"].replace(0, 1)
    print("    + Monthly_Payment")
if "Credit amount" in df.columns and "Age" in df.columns:
    df["Credit_per_Age"] = df["Credit amount"] / (df["Age"] + 1)
    print("    + Credit_per_Age")
if "Saving accounts" in df.columns and "Checking account" in df.columns:
    df["Total_Account_Strength"] = df["Saving accounts"] + df["Checking account"]
    print("    + Total_Account_Strength")


# ── 6. EDA Plots ──────────────────────────────────────────────────────────────
print("\n[6] Generating EDA plots -> outputs/")

target = "Risk"
X_raw = df.drop(columns=[target])
y     = df[target]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("FinGuard AI – Exploratory Data Analysis", fontsize=16, fontweight="bold")

# Class distribution
counts = y.value_counts().sort_index()
labels = [f"Good (0): {counts.get(0,0)}", f"Bad (1): {counts.get(1,0)}"]
axes[0, 0].bar(labels, [counts.get(0,0), counts.get(1,0)],
               color=["#4ade80", "#f87171"], edgecolor="black")
axes[0, 0].set_title("Class Distribution")

# Credit Amount
if "Credit amount" in df.columns:
    axes[0, 1].hist(df["Credit amount"], bins=30, color="#818cf8", edgecolor="black")
    axes[0, 1].set_title("Credit Amount Distribution")
    axes[0, 1].set_xlabel("Credit Amount")

# Age
if "Age" in df.columns:
    axes[1, 0].hist(df["Age"], bins=20, color="#fb923c", edgecolor="black")
    axes[1, 0].set_title("Age Distribution")
    axes[1, 0].set_xlabel("Age")

# Correlation heatmap
corr_cols = X_raw.corrwith(y).abs().nlargest(8).index.tolist()
sns.heatmap(df[corr_cols + [target]].corr(), ax=axes[1, 1],
            annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
axes[1, 1].set_title("Correlation Heatmap (Top 8 Features)")

plt.tight_layout()
plt.savefig("outputs/eda_plots.png", dpi=150)
plt.close()
print("    Saved: outputs/eda_plots.png")


# ── 7. Split & Scale ──────────────────────────────────────────────────────────
print("\n[7] Splitting dataset...")

X = X_raw.copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"    Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ── 8. Train Models ───────────────────────────────────────────────────────────
print("\n[8] Training models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":        DecisionTreeClassifier(random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, random_state=42),
}

results = {}
for name, model in models.items():
    use_sc = name == "Logistic Regression"
    model.fit(X_train_sc if use_sc else X_train,
              y_train)
    y_pred  = model.predict(X_test_sc if use_sc else X_test)
    y_proba = model.predict_proba(X_test_sc if use_sc else X_test)[:, 1]
    results[name] = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred,        zero_division=0), 4),
        "AUC-ROC":   round(roc_auc_score(y_test, y_proba), 4),
        "model_obj": model, "y_pred": y_pred, "y_proba": y_proba,
    }
    print(f"    {name:25s}  Acc:{results[name]['Accuracy']:.4f}  AUC:{results[name]['AUC-ROC']:.4f}")


# ── 9. Hyperparameter Tuning ──────────────────────────────────────────────────
print("\n[9] Tuning Random Forest...")

gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators":[100,200], "max_depth":[None,10,20], "min_samples_split":[2,5]},
    cv=5, scoring="roc_auc", n_jobs=-1)
gs.fit(X_train, y_train)
best_rf      = gs.best_estimator_
y_pred_best  = best_rf.predict(X_test)
y_proba_best = best_rf.predict_proba(X_test)[:, 1]
print(f"    Best params: {gs.best_params_}  CV AUC: {gs.best_score_:.4f}")

results["RF Tuned"] = {
    "Accuracy":  round(accuracy_score(y_test, y_pred_best), 4),
    "Precision": round(precision_score(y_test, y_pred_best, zero_division=0), 4),
    "Recall":    round(recall_score(y_test, y_pred_best,    zero_division=0), 4),
    "F1-Score":  round(f1_score(y_test, y_pred_best,        zero_division=0), 4),
    "AUC-ROC":   round(roc_auc_score(y_test, y_proba_best), 4),
    "model_obj": best_rf, "y_pred": y_pred_best, "y_proba": y_proba_best,
}


# ── 10. Evaluate ──────────────────────────────────────────────────────────────
print("\n[10] Evaluation:")

metrics_df = pd.DataFrame({
    k: {m: v for m, v in v.items() if m not in ["model_obj","y_pred","y_proba"]}
    for k, v in results.items()
}).T
print(metrics_df.to_string())
metrics_df.to_csv("outputs/model_comparison.csv")

best_name = metrics_df["AUC-ROC"].astype(float).idxmax()
best_info  = results[best_name]
print(f"\n    Best model: {best_name}  (AUC={best_info['AUC-ROC']})")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Evaluation – {best_name}", fontsize=14, fontweight="bold")

cm = confusion_matrix(y_test, best_info["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Good","Bad"], yticklabels=["Good","Bad"], ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    axes[1].plot(fpr, tpr, label=f"{name} ({res['AUC-ROC']})")
axes[1].plot([0,1],[0,1],"k--")
axes[1].set_title("ROC Curves")
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
axes[1].legend(fontsize=7)

plt.tight_layout()
plt.savefig("outputs/evaluation_plots.png", dpi=150)
plt.close()
print("    Saved: outputs/evaluation_plots.png")

fi = pd.Series(best_info["model_obj"].feature_importances_, index=X.columns)
fi.nlargest(12).sort_values().plot(kind="barh", color="#6366f1", figsize=(9,5))
plt.title(f"Top 12 Feature Importances – {best_name}", fontweight="bold")
plt.xlabel("Importance"); plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.close()
print("    Saved: outputs/feature_importance.png")


# ── 11. Save Artifacts ────────────────────────────────────────────────────────
print("\n[11] Saving artifacts...")
joblib.dump(best_info["model_obj"], "outputs/finguard_model.pkl")
joblib.dump(scaler,                 "outputs/scaler.pkl")
joblib.dump(list(X.columns),        "outputs/feature_names.pkl")
print("    Saved model, scaler, feature names to outputs/")

print("\n" + "=" * 60)
print("  Pipeline complete! Check the outputs/ folder.")
print("=" * 60)
