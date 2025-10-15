# ===================== lr_transformer_best.py =====================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = r"C:\Users\14217\Desktop\DM2583\project\data_set\Dataset_mapped_final.csv"
MODEL_PATH = r"C:\Users\14217\Desktop\DM2583\project\logistic_transformer_best.pkl"
SAVE_DIR = r"C:\Users\14217\Desktop\DM2583\Logistic Regression\Transformer_Best"
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

transformer_mean_abs = {
    'HighChol': 0.0514,'GenHlth':0.109,'HvyAlcoholConsump':0.0059,'BMI':0.0374,
    'HighBP':0.0648,'Chol_BMI_inter':0.0247,'GenHlth_BMI_inter':0.0137,'CholCheck':0.0168,
    'Veg_BMI_inter':0.0087,'Stroke':0.0023,'PhysActivity':0.0116,'MentHlth':0.0091,
    'Veggies':0.0105,'HeartDiseaseorAttack':0.0155,'PhysAct_BMI_inter':0.0161,'Sex':0.0092,
    'DiffWalk':0.0116,'PhysHlth_BMI_inter':0.0033,'MentHlth_BMI_inter':0.0047,'Fruit_BMI_inter':0.0053,
    'Fruits':0.0028,'BP_Chol_inter':0.0171,'BMI_Walk_inter':0.0069,'PhysHlth':0.0100,'Smoker':0.0043,'BP_BMI_inter':0.0171
}

scaling_factor = 5
for i, f in enumerate(feature_columns):
    if f in transformer_mean_abs:
        X[:, i] = X[:, i] * (1 + scaling_factor * transformer_mean_abs[f])

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
feature_columns_poly = poly.get_feature_names_out(feature_columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)
print(f"Resampled data shape: {X_res.shape}")

weights_dict = {f: transformer_mean_abs.get(f.split("__")[0], 0) for f in feature_columns_poly}
init_weights = np.array([weights_dict.get(f, 0) for f in feature_columns_poly])
init_intercept = np.log(y_res.mean()/(1-y_res.mean()))

model = LogisticRegression(max_iter=3000, solver='lbfgs', class_weight='balanced')
model.fit(X_res, y_res)
model.coef_ = init_weights.reshape(1, -1)
model.intercept_ = np.array([init_intercept])
model.fit(X_res, y_res)

joblib.dump(model, MODEL_PATH)

y_pred = model.predict(X_res)
y_proba = model.predict_proba(X_res)[:,1]

print("\nClassification Report:")
print(classification_report(y_res, y_pred))

cm = confusion_matrix(y_res, y_pred)
print("Confusion Matrix:\n", cm)

auc = roc_auc_score(y_res, y_proba)
print(f"AUC: {auc:.4f}")

RocCurveDisplay.from_estimator(model, X_res, y_res)
plt.title("ROC Curve - LR Enhanced Transformer Best")
plt.savefig(os.path.join(SAVE_DIR, "ROC_Curve.png"), dpi=300, bbox_inches='tight')
plt.close()

print("\nFinal trained weights (coef):")
for f, w in zip(feature_columns_poly, model.coef_[0]):
    print(f"{f}: {w:.6f}")
print(f"Intercept: {model.intercept_[0]:.6f}")

print(f"\nModel saved to {MODEL_PATH}")
