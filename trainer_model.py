import r2_download as hd
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
LABEL_COL = 'heavy_rain_6h'   # you can also use 'heavy_rain_3h_jun' etc.
SEQ_LENGTH = 48*2            # 12 hours of 15‑min data (adjust as needed)

train_dataset = WeatherDataset(train_df, FEATURE_COLS, ALL_LABELS, SEQ_LENGTH)
val_dataset   = WeatherDataset(val_df,   FEATURE_COLS, ALL_LABELS, SEQ_LENGTH)
test_dataset  = WeatherDataset(test_df,  FEATURE_COLS, ALL_LABELS, SEQ_LENGTH)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

batch_size = 1028
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          drop_last=True, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                          drop_last=False, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                          drop_last=False, pin_memory=True)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ----------------------------------------------------------------------
# 5. Model definitions (unchanged from your code)
# ----------------------------------------------------------------------
class RecurrentClassifier(nn.Module):
    """Shared architecture for RNN/LSTM/GRU binary classifiers."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=6, dropout=0.3,
                 rnn_type='lstm'):
        super().__init__()
        self.rnn_type = rnn_type

        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        output, _ = self.rnn(x)
        last_hidden = output[:, -1, :]  # (batch, hidden_dim)
        logit = self.classifier(last_hidden)  # (batch, 1)
        return logit.squeeze(-1)  # (batch,)

# ----------------------------------------------------------------------
# 6. Instantiate all three models
# ----------------------------------------------------------------------
n_features = len(FEATURE_COLS)
models = {
    #'RNN': (RecurrentClassifier(n_features, rnn_type='rnn').to(device), False),
    #'LSTM': (RecurrentClassifier(n_features, rnn_type='lstm').to(device), False),
    # 'GRU': (RecurrentClassifier(n_features, rnn_type='gru').to(device), False),
    #based on input_dim, T_in, T_out, latent_dim=16, hidden_dim=128, dropout=0.3
    'causalVAE': (CausalVAE(input_dim=n_features, T_in=SEQ_LENGTH, T_out=len(ALL_LABELS), latent_dim=16, hidden_dim=512, dropout=0.3).to(device)),
    # 'DeeperCausalVAE' : (DeeperCausalVAE(input_dim=n_features, T_in=SEQ_LENGTH, T_out=len(ALL_LABELS), latent_dim=16, hidden_dim=512, dropout=0.3).to(device)),
  }

for name, (model) in models.items():
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_params:,} parameters")

# ----------------------------------------------------------------------
# 7. Training function (unchanged, except we pass the loaders)
# ----------------------------------------------------------------------
def compute_class_weight(dataset):
    """Compute positive class weight from dataset labels."""
    labels = dataset.labels[dataset.valid_indices]
    pos_rate = labels.mean()
    if pos_rate == 0 or pos_rate == 1:
        return 1.0
    weight = (1 - pos_rate) / pos_rate
    return min(weight, 20.0)

def train_model(model, train_loader, val_loader, pos_weight,
                lr=1e-4, max_epochs=50, patience=10, model_name='Model_smaller'):
    """Train a model with early stopping on validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience
    )

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'recon_x' : [], 'recon_y':[], 'kl_loss': [], 'val_prediction': []}

    pbar = tqdm(range(max_epochs), desc=f"Training {model_name}")
    for epoch in pbar:
        # === Train ===
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            loss, (recon_loss_x, recon_loss_y, kl_loss) = model(x_batch, y_batch)
            history['recon_x'].append(recon_loss_x.item())
            history['recon_y'].append(recon_loss_y.item())
            history['kl_loss'].append(kl_loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Recon X Loss={np.mean(history['recon_x']):.4f}, ")
            print(f"Epoch {epoch}: Recon Y Loss={np.mean(history['recon_y']):.4f}, ")
            print(f"Epoch {epoch}: KL Loss={np.mean(history['kl_loss']):.4f}, ")

        # === Validate ===
        model.eval()
        val_losses = []
        for x_batch, y_batch in val_loader:
            batch_size = x_batch.size(0)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device) 
            
            # Target shape: (1, batch_size, 1, 1)
            y_target = y_batch.view(1, batch_size, -1, 1) 
            # Generate deterministic mean! Output shape: (1, batch_size, 1, 1)
            y_pred_mean = model.generate(x_batch, deterministic=True) 
            # Calculate stable loss (no expanding needed)
            loss = criterion(y_pred_mean, y_target)
            
            val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        # === Early stopping ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Restore best model
    model.load_state_dict(best_state)
    return history
# ----------------------------------------------------------------------
# 8. Train each model (example)
# ----------------------------------------------------------------------
pos_weight = compute_class_weight(train_dataset)
print(f"Positive class weight: {pos_weight:.3f}")


histories = {}
for name, (model) in models.items():
    print(f"\n{'='*50}")
    print(f"  Training {name}")
    print(f"{'='*50}")
    histories[name] = train_model(
        model, train_loader, val_loader,
        pos_weight=pos_weight,
        lr=1e-3,
        max_epochs=50,
        patience=20,
        model_name=name + '_smaller',
    )

# === Save checkpoints ===
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

for name, model in models.items():
    name = name + ''
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{name.lower()}_heavy_rain_all_1day.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'rnn_type': name.lower(),
        'input_dim': n_features,
        'hidden_dim': 512,
        'dropout': 0.3,
        'target': TARGET_STATION,
        'lookback': "LOOKBACK",
    }, ckpt_path)
    print(f"  Saved {name} checkpoint → {ckpt_path}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, hist) in zip(axes, histories.items()):
    ax.plot(hist['train_loss'], label='Train')
    ax.plot(hist['val_loss'], label='Val')
    ax.set_title(f'{name} — Loss Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
plt.tight_layout()
plt.savefig("training_loss_curves_model_smaller.png")
plt.show()

# === Plot predictions vs ground truth after training ===
print("\nGenerating predictions vs ground truth plot...")
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_pred = model.generate(x_batch, deterministic=True)
        # Reshape predictions to match targets
        y_pred = y_pred.squeeze()  # Remove extra dimensions
        all_predictions.extend(y_pred.cpu().numpy())
        all_targets.extend(y_batch.numpy())

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

# # Balanced accuracy check
# threshold = 0.5
# y_pred_labels = (all_predictions >= threshold).astype(int)
# balanced_acc = balanced_accuracy_score(all_targets, y_pred_labels)
# print(f"Balanced Accuracy at threshold {threshold}: {balanced_acc:.4f}")

