import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from data_processor import DataProcessor
from model import DiabetesTransformer

CONFIG = {
    "DATA_PATH": r"C:\Users\14217\Desktop\DM2583\project\data_set\Dataset_mapped_final.csv",
    "BATCH_SIZE": 128,
    "EPOCHS": 80,
    "LEARNING_RATE": 7e-4,
    "WEIGHT_DECAY": 1e-4,
    "PATIENCE": 15,
    "D_MODEL": 128,
    "NHEAD": 8,
    "NUM_LAYERS": 3,
    "DROPOUT": 0.3,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "TEST_SIZE": 0.2,
    "VAL_SIZE": 0.1,
    "RANDOM_STATE": 42,
    "MODEL_PATH": "best_transformer.pth"
}

class Trainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.train_losses, self.val_losses = [], []
        self.train_aucs, self.val_aucs = [], []

    def train_epoch(self, loader, criterion, optimizer):
        self.model.train()
        total_loss, all_preds, all_labels = 0, [], []
        for features, labels in loader:
            features, labels = features.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        return total_loss / len(loader), accuracy_score(all_labels, all_preds)

    def evaluate(self, loader, criterion):
        self.model.eval()
        total_loss, all_preds, all_labels, all_probs = 0, [], [], []
        with torch.no_grad():
            for features, labels in loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(probs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
        return total_loss / len(loader), accuracy_score(all_labels, all_preds), auc, all_preds, all_labels

    def train(self, train_loader, val_loader, test_loader, config):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = optim.AdamW(self.model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
        scheduler = CosineAnnealingLR(optimizer, T_max=config["EPOCHS"])
        best_val_auc, patience_counter = 0, 0

        for epoch in range(config["EPOCHS"]):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc, val_auc, _, _ = self.evaluate(val_loader, criterion)
            scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_aucs.append(train_acc)
            self.val_aucs.append(val_auc)

            print(f"Epoch [{epoch+1}/{config['EPOCHS']}] | LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Time: {time.time()-start_time:.1f}s")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), config["MODEL_PATH"])
            else:
                patience_counter += 1
            if patience_counter >= config["PATIENCE"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self.model.load_state_dict(torch.load(config["MODEL_PATH"]))
        test_loss, test_acc, test_auc, test_preds, test_labels = self.evaluate(test_loader, criterion)
        print("\nFinal Test Results:")
        print(f"Test loss={test_loss:.4f} | Test Acc={test_acc:.4f} | Test AUC={test_auc:.4f}")
        print(classification_report(test_labels, test_preds, target_names=['No Diabetes','Diabetes']))

        self.plot_training_curve()

    def plot_training_curve(self):
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training / Validation Loss")
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(self.train_aucs, label="Train Acc")
        plt.plot(self.val_aucs, label="Val AUC")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Training Accuracy / Validation AUC")
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    device = CONFIG["DEVICE"]
    print(f"Using device: {device}")

    processor = DataProcessor(CONFIG)
    X, y, feature_columns = processor.load_and_process_data(CONFIG["DATA_PATH"])
    train_loader, val_loader, test_loader = processor.create_data_loaders(X, y, batch_size=CONFIG["BATCH_SIZE"])

    model = DiabetesTransformer(num_features=X.shape[1], config=CONFIG).to(device)
    trainer = Trainer(model, device)
    trainer.train(train_loader, val_loader, test_loader, CONFIG)

if __name__ == "__main__":
    main()
