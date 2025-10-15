import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from train import DiabetesTransformer, DataProcessor, CONFIG

def shap_analysis(model, X, feature_names, device="cuda", batch_size=500, max_samples=None):
    model.eval()
    model.to(device)
    print(f" Using device: {device}")

    def model_predict(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1]
        return probs.cpu().numpy()

    if max_samples is not None:
        X = X[:max_samples]
        print(f" Using first {X.shape[0]} samples for SHAP analysis.")

    explainer = shap.Explainer(model_predict, X[:batch_size])

    shap_values_list = []
    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i+batch_size]
        shap_values_batch = explainer(X_batch)
        shap_values_list.append(shap_values_batch.values)

    shap_values_all = np.vstack(shap_values_list)

    mean_shap = shap_values_all.mean(axis=0)
    mean_abs_shap = np.abs(shap_values_all).mean(axis=0)
    feature_importance = sorted(zip(feature_names, mean_shap, mean_abs_shap),
                                key=lambda x: abs(x[1]), reverse=True)

    print("\nFeature ranking by mean SHAP value:")
    for name, mean_val, abs_val in feature_importance:
        tag = "(interaction)" if "inter" in name else ""
        print(f"{name}: mean={mean_val:.4f}, mean_abs={abs_val:.4f} {tag}")

    print("Generating SHAP summary plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_all, features=X, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot (All Features)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Generating Top 10 feature bar chart...")
    top_n = 10
    top_features = feature_importance[:top_n]
    top_names = [f for f, _, _ in top_features][::-1]
    top_means = [m for _, m, _ in top_features][::-1]

    plt.figure(figsize=(10, 8))
    colors = ['red' if v > 0 else 'blue' for v in top_means]
    bars = plt.barh(range(len(top_names)), top_means, color=colors, edgecolor='black', height=0.7, alpha=0.8)
    plt.yticks(range(len(top_names)), top_names, fontsize=11)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Mean SHAP Value", fontsize=12)
    plt.title("Top 10 Features Impact\n(Red: Increase Risk, Blue: Decrease Risk)", fontsize=14, pad=20)
    for i, (bar, value) in enumerate(zip(bars, top_means)):
        width = bar.get_width()
        if value >= 0:
            x_pos = width + 0.001
            ha = 'left'
        else:
            x_pos = +0.001
            ha = 'right'
        plt.text(x_pos, i, f'{value:.4f}', ha=ha, va='center', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("shap_top10_features.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Generating all-features SHAP bar chart...")
    feature_df = pd.DataFrame({
        "Feature": [f for f, _, _ in feature_importance],
        "Mean_SHAP": [m for _, m, _ in feature_importance]
    })

    plt.figure(figsize=(12, 14))
    colors = ['red' if v > 0 else 'blue' for v in feature_df["Mean_SHAP"]]
    bars = plt.barh(feature_df["Feature"], feature_df["Mean_SHAP"], color=colors, edgecolor='black', height=0.8, alpha=0.7)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.gca().invert_yaxis()
    plt.title("All Features Mean SHAP Values (Direction-aware)", fontsize=16, pad=20)
    plt.xlabel("Mean SHAP Value", fontsize=12)
    for i, (bar, value) in enumerate(zip(bars[:15], feature_df["Mean_SHAP"][:15])):
        width = bar.get_width()
        if value >= 0:
            x_pos = width + 0.0005
            ha = 'left'
        else:
            x_pos = 0.0005
            ha = 'left'
        plt.text(x_pos, i, f'{value:.4f}', va='center', ha=ha,
                 fontsize=8, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("shap_all_features_bar.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Generating SHAP feature interaction heatmap...")
    shap_corr = np.corrcoef(shap_values_all.T)
    plt.figure(figsize=(14, 12))
    im = plt.imshow(shap_corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, label="SHAP Value Correlation", shrink=0.8)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90, fontsize=8)
    plt.yticks(range(len(feature_names)), feature_names, fontsize=8)
    threshold = 0.3
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            if abs(shap_corr[i, j]) > threshold and i != j:
                plt.text(j, i, f'{shap_corr[i, j]:.2f}', ha="center", va="center", fontsize=6,
                         fontweight='bold' if abs(shap_corr[i, j]) > 0.5 else 'normal')
    plt.title("SHAP Feature Interaction Heatmap", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("shap_interaction_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Generating SHAP dependence plots...")
    top5 = [f for f, _, _ in feature_importance[:5]]
    for feat in top5:
        feat_idx = feature_names.index(feat)
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feat_idx, shap_values_all, X, feature_names=feature_names, show=False)
        plt.title(f"SHAP Dependence Plot: {feat}", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f"shap_dependence_{feat.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.close()

    top3 = feature_importance[:3]
    print("\nModel Interpretation Summary:")
    for f, m, _ in top3:
        direction = "increases diabetes risk" if m > 0 else "decreases diabetes risk"
        tag = "(interaction)" if "inter" in f else ""
        print(f" - {f}{tag}: mean contribution {m:.4f}, tends to {direction}.")

    print("\n SHAP visualizations generated:")
    print("   - shap_summary_plot.png")
    print("   - shap_top10_features.png")
    print("   - shap_all_features_bar.png")
    print("   - shap_interaction_heatmap.png")
    print("   - shap_dependence_<feature>.png")

if __name__ == "__main__":
    processor = DataProcessor(CONFIG)
    X, y, feature_columns = processor.load_and_process_data(CONFIG["DATA_PATH"])
    device = CONFIG["DEVICE"]
    model = DiabetesTransformer(num_features=X.shape[1], config=CONFIG).to(device)
    model.load_state_dict(torch.load(
        r"C:\Users\14217\Desktop\DM2583\best_transformer.pth",
        map_location=device
    ))
    shap_analysis(
        model=model,
        X=X,
        feature_names=feature_columns,
        device=device,
        batch_size=500,
        max_samples=13923
    )
