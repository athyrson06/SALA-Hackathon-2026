import torch
import r2_download as hd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import seaborn as sns
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F

import pandas as pd
import numpy as np
import tqdm

import warnings
warnings.filterwarnings('ignore')

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import timedelta
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    balanced_accuracy_score, precision_recall_curve, average_precision_score, roc_auc_score,
    confusion_matrix, classification_report, brier_score_loss, f1_score
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from conditional_vae import CausalVAE

import netCDF4

# === Reproducibility ===
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = hd._default_data_dir()
PRECIP_DIR = f"{DATA_DIR}/precipitation-nowcasting"

# === Station info (must match preprocessing) ===
STATION_FILES = {
    'cer': 'weather_stations/CER_consolid_f15.csv',
    'jun': 'weather_stations/JUN_consolid_f15.csv',
    'merc': 'weather_stations/MERC_consolid_f15.csv',
    'mira': 'weather_stations/MIRA_consolid_f15.csv',
}
STATIONS = list(STATION_FILES.keys())
TARGET_STATION = 'jun'
HORIZONS = {'3h': 12, '6h': 24, '12h': 48}  # steps at 15-min resolution

# === Labels that were created in preprocessing (same pattern) ===
LABEL_COLS = [f'heavy_rain_{h}' for h in HORIZONS]               # target station
ALL_LABELS = []
for stn in STATIONS:
    LABEL_COLS += [f'heavy_rain_{h}_{stn}' for h in HORIZONS]    # all stations
    ALL_LABELS += [f'heavy_rain_{h}_{stn}' for h in HORIZONS]
LABEL_COLS += ['temp_extreme', 'temp_anomaly']


# ----------------------------------------------------------------------
# 1. Load the pre‑processed data (saved by the first script)
# ----------------------------------------------------------------------
print("Loading pre‑processed data...")
train_df = pd.read_csv(f"{PRECIP_DIR}/data_processed/train_data.csv",
                       index_col=0, parse_dates=True)
val_df   = pd.read_csv(f"{PRECIP_DIR}/data_processed/val_data.csv",
                       index_col=0, parse_dates=True)
test_df  = pd.read_csv(f"{PRECIP_DIR}/data_processed/test_data.csv",
                       index_col=0, parse_dates=True)


print(f"Train: {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")
print(f"Val:   {val_df.shape[0]:,} rows, {val_df.shape[1]} columns")
print(f"Test:  {test_df.shape[0]:,} rows, {test_df.shape[1]} columns")

# ----------------------------------------------------------------------
# 2. Define feature columns (all columns except the labels)
# ----------------------------------------------------------------------
FEATURE_COLS = [c for c in train_df.columns if c not in LABEL_COLS]
print(f"Number of features: {len(FEATURE_COLS)}")

# ----------------------------------------------------------------------
# 3. PyTorch Dataset for time‑series sequences
# ----------------------------------------------------------------------
class WeatherDataset(Dataset):
    """Sliding window dataset for weather time series classification."""

    def __init__(self, df, feature_cols, target_col, lookback=96):
        self.lookback = lookback

        # === Drop rows with NaN in features or target ===
        if isinstance(target_col, list):
            all_cols = feature_cols + target_col
        else:            
            all_cols = feature_cols + [target_col]
        valid_mask = df[all_cols].notna().all(axis=1)
        clean_df = df.loc[valid_mask].copy()

        self.features = clean_df[feature_cols].values.astype(np.float32)
        self.labels = clean_df[target_col].values.astype(np.float32)
        self.timestamps = clean_df.index

        # === Build valid window indices ===
        # A window is valid if all `lookback` consecutive rows are from consecutive
        # 15-min timestamps (no time gaps within the window)
        self.valid_indices = []
        expected_delta = pd.Timedelta(minutes=15)
        for i in tqdm(range(lookback, len(self.features)),
                      desc=f"Building windows ({target_col})", leave=False):
            # Check that timestamps are consecutive within the window
            window_times = self.timestamps[i - lookback:i + 1]
            diffs = window_times[1:] - window_times[:-1]
            if (diffs == expected_delta).all():
                self.valid_indices.append(i)

        self.valid_indices = np.array(self.valid_indices)
        print(f"  {target_col}: {len(self.valid_indices):,} valid windows "
              f"from {len(self.features):,} rows "
              f"(positive rate: {self.labels[self.valid_indices].mean():.3%})")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        x = self.features[i - self.lookback:i]  # (lookback, n_features)
        y = self.labels[i]                        # scalar
        return torch.from_numpy(x), torch.tensor(y)

# ----------------------------------------------------------------------
# 4. Choose the label we want to predict (e.g. 3h heavy rain at target station)
# ----------------------------------------------------------------------
LABEL_COL = 'heavy_rain_3h'   # you can also use 'heavy_rain_3h_jun' etc.
SEQ_LENGTH = 48             # 12 hours of 15‑min data (adjust as needed)

test_dataset  = WeatherDataset(test_df,  FEATURE_COLS, ALL_LABELS, SEQ_LENGTH)

batch_size = 512
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                          drop_last=False, pin_memory=True)

# Path to your trained model
MODEL_PATH = "checkpoints/causalvae_heavy_rain_all.pt"  # Update this
n_features = len(FEATURE_COLS)
# Model parameters (must match training)
MODEL_PARAMS = {
    'input_dim': n_features,     # Number of features per time step
    'T_in': SEQ_LENGTH,          # Input sequence length
    'T_out': len(ALL_LABELS),         # Output sequence length
    'latent_dim': 16,    # Latent dimension
    'hidden_dim': 512    # Hidden dimension
}

def evaluate_causal_vae(model_path, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load trained model and evaluate on test data.
    
    Args:
        model_path: Path to the saved model (.pth file)
        test_loader: DataLoader for test data
        device: Device to run inference on
    """
    
    # 1. Load the Model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # If you saved the entire model
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Recreate model with same parameters (you'll need to know these)
        # Adjust these values to match your training configuration
        model = CausalVAE(
            input_dim=MODEL_PARAMS['input_dim'],      # Update with your actual input_dim
            T_in=MODEL_PARAMS['T_in'],           # Update with your actual T_in
            T_out=MODEL_PARAMS['T_out'],          # Update with your actual T_out
            latent_dim=MODEL_PARAMS['latent_dim'],     # Update with your actual latent_dim
            hidden_dim=MODEL_PARAMS['hidden_dim']     # Update with your actual hidden_dim
        )
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If you saved the whole model directly
        model = checkpoint

    
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    print(f"Using device: {device}")
    
    # 2. Run Inference
    all_predictions = []
    all_targets = []
    all_recon_errors = []
    
    print("Generating predictions...")
    
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Get predictions
            y_pred = model.generate(x_batch, deterministic=True)  # Shape: (1, batch, T_out, 1)
            y_pred = y_pred.squeeze(0).squeeze(-1)  # Shape: (batch, T_out)
            
            # Get reconstruction for quality check
            _, Z = model.encode(x_batch, y_batch)  # You might need to modify encode to return Z
            x_recon = model.decode_x(Z)
            recon_error = F.mse_loss(x_recon, x_batch, reduction='none').mean(dim=(1,2))
            
            all_predictions.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())
            all_recon_errors.append(recon_error.cpu())
            
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx * len(x_batch)} samples")
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0).numpy()  # Shape: (n_samples, T_out)
    targets = torch.cat(all_targets, dim=0).numpy()  # Shape: (n_samples, T_out)
    recon_errors = torch.cat(all_recon_errors, dim=0).numpy()
    
    # Flatten for metrics (treat each time step independently)
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Convert logits to probabilities
    probs_flat = 1 / (1 + np.exp(-pred_flat))
    pred_binary = (probs_flat > 0.5).astype(int)
    
    print(f"\nTotal samples: {len(targets)}")
    print(f"Positive rate: {target_flat.mean():.4f}")
    
    # 3. Calculate Metrics
    print("\n" + "="*50)
    print("METRICS")
    print("="*50)
    
    # Classification metrics
    accuracy = accuracy_score(target_flat, pred_binary)
    precision = precision_score(target_flat, pred_binary, zero_division=0)
    recall = recall_score(target_flat, pred_binary, zero_division=0)
    f1 = f1_score(target_flat, pred_binary, zero_division=0)
    
    # Probabilistic metrics
    try:
        auc_roc = roc_auc_score(target_flat, probs_flat)
        auc_pr = average_precision_score(target_flat, probs_flat)
    except:
        auc_roc = 0.5
        auc_pr = target_flat.mean()
        print("Warning: Could not compute AUC scores")
    
    print(f"\nBinary Classification Metrics (threshold=0.5):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nProbabilistic Metrics:")
    print(f"  AUC-ROC:   {auc_roc:.4f}")
    print(f"  AUC-PR:    {auc_pr:.4f}")
    
    print(f"\nReconstruction Quality:")
    print(f"  Mean Recon Error: {recon_errors.mean():.4f}")
    print(f"  Std Recon Error:  {recon_errors.std():.4f}")
    
    # 4. Confusion Matrix
    cm = confusion_matrix(target_flat, pred_binary)
    breakpoint()
    
    # 5. Visualizations
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Calibration Curve
    ax1 = plt.subplot(3, 3, 1)
    prob_true, prob_pred = calibration_curve(target_flat, probs_flat, n_bins=10)
    ax1.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    ax1.plot([0, 1], [0, 1], 'r--', label='Perfect')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('Calibration Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Confusion Matrix
    ax2 = plt.subplot(3, 3, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix')
    
    # Plot 3: Prediction Distribution
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(probs_flat[target_flat==0], bins=30, alpha=0.5, label='Negative', density=True)
    ax3.hist(probs_flat[target_flat==1], bins=30, alpha=0.5, label='Positive', density=True)
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Density')
    ax3.set_title('Prediction Distribution by Class')
    ax3.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig("images/evaluation_results.png")
    
    # Save results to file
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'mean_recon_error': recon_errors.mean(),
        'predictions': predictions,
        'targets': targets,
        'probabilities': probs_flat.reshape(targets.shape)
    }
    
    return results


if __name__ == "__main__":
    # ==================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ==================================================
    
    # Data parameters
    BATCH_SIZE = 32
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ==================================================
    # RUN EVALUATION
    # ==================================================
    
    print("Starting evaluation...")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    # Run evaluation
    results = evaluate_causal_vae(MODEL_PATH, test_loader, DEVICE)
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print("Key metrics:")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")