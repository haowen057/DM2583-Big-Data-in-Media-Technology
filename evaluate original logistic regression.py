import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = r"C:\Users\14217\Desktop\DM2583\project\data_set\Dataset_mapped_final.csv"
MODEL_PATH = r"C:\Users\14217\Desktop\DM2583\project\logistic_model_combinations.pkl"
SAVE_DIR = r"C:\Users\14217\Desktop\DM2583\Logistic Regression"
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Data dimensions: {df.shape}")

feature_columns = [
    'HighBP','HighChol','BMI','Stroke','HeartDiseaseorAttack','DiffWalk','Sex',
    'CholCheck','Smoker','PhysActivity','Fruits','Veggies','HvyAlcoholConsump',
    'GenHlth','MentHlth','PhysHlth'
]

df['BMI_Walk_inter'] = df['BMI'] * df['DiffWalk']
df['BP_Chol_inter'] = df['HighBP'] * df['HighChol']
df['BP_BMI_inter'] = df['HighBP'] * df['BMI']
df['Chol_BMI_inter'] = df['HighChol'] * df['BMI']
df['PhysAct_BMI_inter'] = df['PhysActivity'] * df['BMI']
df['Fruit_BMI_inter'] = df['Fruits'] * df['BMI']
df['Veg_BMI_inter'] = df['Veggies'] * df['BMI']
df['MentHlth_BMI_inter'] = df['MentHlth'] * df['BMI']
df['PhysHlth_BMI_inter'] = df['PhysHlth'] * df['BMI']
df['GenHlth_BMI_inter'] = df['GenHlth'] * df['BMI']

new_features = [
    'BMI_Walk_inter','BP_Chol_inter','BP_BMI_inter','Chol_BMI_inter',
    'PhysAct_BMI_inter','Fruit_BMI_inter','Veg_BMI_inter',
    'MentHlth_BMI_inter','PhysHlth_BMI_inter','GenHlth_BMI_inter'
]
feature_columns += new_features

X = df[feature_columns].values
y = df['Diabetes_01'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)
print(f"Sample Enhanced Dimension: {X_res.shape}")

retrain = True
if os.path.exists(MODEL_PATH) and not retrain:
    print("Loading pre-trained model...")
    model = joblib.load(MODEL_PATH)
else:
    print("Training Logistic Regression model (26 features)...")
    model = LogisticRegression(max_iter=2000)
    model.fit(X_res, y_res)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

y_pred = model.predict(X_res)
y_proba = model.predict_proba(X_res)[:, 1]

print("\nClassification Report:")
print(classification_report(y_res, y_pred))

cm = confusion_matrix(y_res, y_pred)
print("Confusion Matrix:\n", cm)

auc = roc_auc_score(y_res, y_proba)
print(f"AUC: {auc:.4f}")

RocCurveDisplay.from_estimator(model, X_res, y_res)
plt.title("ROC Curve - Logistic Regression (AUC = {:.4f})".format(auc))
plt.savefig(os.path.join(SAVE_DIR, "ROC_Curve.png"), dpi=300, bbox_inches="tight")
plt.close()

weights = model.coef_.flatten()
bias = model.intercept_[0]

print("\nModel Weights (coef):")
for f, w in zip(feature_columns, weights):
    print(f"{f}: {w:.6f}")
print(f"Intercept: {bias:.6f}")

weights_df = pd.DataFrame({
    "Feature": feature_columns,
    "Weight": weights
})
weights_df.to_csv(os.path.join(SAVE_DIR, "model_weights.csv"), index=False)
print(f"\nModel weights saved to {os.path.join(SAVE_DIR, 'model_weights.csv')}")

print("Starting SHAP analysis...")
explainer = shap.Explainer(model, X_res, feature_names=feature_columns)
shap_values = explainer(X_res)
shap_matrix = shap_values.values

print("Generating SHAP Summary Plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_matrix, X_res, feature_names=feature_columns, show=False)
plt.title("SHAP Summary Plot (All Features)", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
plt.close()

mean_shap = shap_matrix.mean(axis=0)
feature_importance = sorted(zip(feature_columns, mean_shap), key=lambda x: abs(x[1]), reverse=True)
top_features = feature_importance[:10]
top_names = [f for f, _ in top_features][::-1]
top_means = [m for _, m in top_features][::-1]

plt.figure(figsize=(10, 8))
colors = ['red' if v > 0 else 'blue' for v in top_means]
plt.barh(range(len(top_names)), top_means, color=colors, edgecolor='black', height=0.7, alpha=0.8)
plt.yticks(range(len(top_names)), top_names, fontsize=11)
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Mean SHAP Value", fontsize=12)
plt.title("Logistic Regression Top 10 Features Impact\n(Red: Increase Risk, Blue: Decrease Risk)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "shap_top10_features.png"), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 14))
colors = ['red' if v > 0 else 'blue' for v in mean_shap]
plt.barh(feature_columns, mean_shap, color=colors, edgecolor='black', height=0.8, alpha=0.7)
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.gca().invert_yaxis()
plt.title("Logistic Regression All Features Mean SHAP Values", fontsize=16, pad=20)
plt.xlabel("Mean SHAP Value", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "shap_all_features_bar.png"), dpi=300, bbox_inches='tight')
plt.close()

top5 = [f for f, _ in feature_importance[:5]]
for feat in top5:
    feat_idx = feature_columns.index(feat)
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feat_idx, shap_matrix, X_res, feature_names=feature_columns, show=False)
    plt.title(f"SHAP Dependence Plot: {feat}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"shap_dependence_{feat.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    plt.close()

print("\nAll SHAP visualizations generated")
print(f"Output Directory: {SAVE_DIR}")
print(f"Training Samples: {X_res.shape[0]}, Feature Count: {X_res.shape[1]}")
