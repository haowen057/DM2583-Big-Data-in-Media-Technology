import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class DiabetesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DataProcessor:
    def __init__(self, config):
        self.test_size = config["TEST_SIZE"]
        self.val_size = config["VAL_SIZE"]
        self.random_state = config["RANDOM_STATE"]
        self.scaler = StandardScaler()

    def load_and_process_data(self, file_path):
        df = pd.read_csv(file_path)
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
        X_scaled = self.scaler.fit_transform(X)
        sm = SMOTE(random_state=self.random_state)
        X_res, y_res = sm.fit_resample(X_scaled, y)
        return X_res, y_res, feature_columns

    def create_data_loaders(self, X, y, batch_size):
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=self.random_state, stratify=y_temp
        )
        train_loader = DataLoader(DiabetesDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(DiabetesDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(DiabetesDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
