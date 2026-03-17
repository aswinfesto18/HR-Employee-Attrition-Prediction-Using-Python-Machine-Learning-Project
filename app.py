import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

sns.set(style="whitegrid")


# Title

st.title("💼 Employee Attrition Prediction")
st.write("""
This app trains Logistic Regression and Random Forest models to predict employee attrition
and visualizes performance metrics and feature importances.
""")

# Load CSV

df = pd.read_csv("HR_Employee_Attrition.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# Preprocessing

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training

st.subheader("Model Training")

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Accuracy & Classification Report

st.write("### Logistic Regression")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.2f}")
st.text(classification_report(y_test, y_pred_lr))

st.write("### Random Forest")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
st.text(classification_report(y_test, y_pred_rf))

# Confusion Matrix

st.write("### Confusion Matrix (Logistic Regression)")
cm = confusion_matrix(y_test, y_pred_lr)
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig_cm)

# ROC Curve

st.write("### ROC Curve (Logistic Regression)")
fpr, tpr, _ = roc_curve(y_test, y_prob_lr)
roc_auc = auc(fpr, tpr)

fig_roc, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax2.plot([0,1], [0,1], linestyle="--")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()
st.pyplot(fig_roc)

# Feature Importance

st.write("### Top 10 Feature Importances (Random Forest)")
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False).head(10)

fig_fi, ax3 = plt.subplots(figsize=(7,4))
sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax3)
st.pyplot(fig_fi)
