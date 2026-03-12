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
import tqdm

import warnings
warnings.filterwarnings('ignore')

import matplotlib.dates as mdates
from datetime import timedelta

import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import (
    balanced_accuracy_score, precision_recall_curve, classification_report, brier_score_loss
)
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
        for i in tqdm.tqdm(range(lookback, len(self.features)),
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

def load_model_and_data(model_path, test_loader, n_features, seq_length):
    """Load trained model and prepare data for interventions."""
    
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Adjust these values to match your training configuration
    model = CausalVAE(
        input_dim=MODEL_PARAMS['input_dim'],      
        T_in=MODEL_PARAMS['T_in'],           
        T_out=MODEL_PARAMS['T_out'],          
        latent_dim=MODEL_PARAMS['latent_dim'],     
        hidden_dim=MODEL_PARAMS['hidden_dim']     
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Get a batch of test data
    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    
    print(f"Loaded {len(x_batch)} samples for intervention")
    
    return model, x_batch, y_batch

def intervene_on_feature(x, feature_idx, intervention_value, time_steps=None):
    """
    Intervene on a specific feature at specific time steps.
    """
    x_intervened = x.clone()
    
    if time_steps is None:
        # Intervene on all time steps
        x_intervened[:, :, feature_idx] = intervention_value
    else:
        # Intervene on specific time steps
        for t in time_steps:
            x_intervened[:, t, feature_idx] = intervention_value
    
    return x_intervened

def run_intervention_experiment(model, x_original, y_original, 
                                feature_name, feature_idx,
                                intervention_values, n_samples=100):
    with torch.no_grad():
        baseline_pred = model.generate(x_original, deterministic=True)
        baseline_pred = baseline_pred.squeeze().cpu().numpy()
        
        # Get baseline latent codes (Z) to condition on
        _, z_original = model.encode(x_original, y_original)
        
        results = {
            'feature': feature_name,
            'intervention_values': intervention_values,
            'baseline_pred': baseline_pred,
            'baseline_latent': z_original.cpu().numpy(),
            'intervened_pred': [],
            'intervened_latent': [],
            'p_y_given_do': [],
            'p_y_given_do_z': []  # <--- NEW: Store P(Y | do(x), z)
        }
        
        for value in tqdm.tqdm(intervention_values, desc=f"Intervening on {feature_name}"):
            x_intervened = intervene_on_feature(x_original, feature_idx, value)
            
            y_pred = model.generate(x_intervened, deterministic=True)
            results['intervened_pred'].append(y_pred.squeeze().cpu().numpy())
            
            _, z_intervened = model.encode(x_intervened, y_original)
            results['intervened_latent'].append(z_intervened.cpu().numpy())
            
            # 1. Standard P(Y | do(X=x))
            y_samples = [model.generate(x_intervened, deterministic=False).squeeze().cpu().numpy() for _ in range(n_samples)]
            results['p_y_given_do'].append(np.mean(np.array(y_samples) > 0.5))

            # ---------------------------------------------------------
            # NEW: Calculate P(Y | do(X=x), Z=z)
            # ---------------------------------------------------------
            # You can either sample z_sample = torch.randn_like(z_original) 
            # OR use the baseline z_original. Here we use the fixed baseline Z.
            # *NOTE*: Adjust `model.decode` to match your CausalVAE's exact method signature!

            y_given_z = model.decode_y(x_intervened, z_original) 
            y_given_z = torch.sigmoid(y_given_z)
            # Bernoulli sample
            y_given_z = torch.bernoulli(y_given_z).unsqueeze(-1).cpu().numpy()
            p_positive_z = np.mean(y_given_z > 0.5) 
                
            # y_given_z = y_given_z.squeeze().cpu().numpy()
            # If your decoder outputs logits, apply sigmoid. If probabilities, just check > 0.5.
            # p_positive_z = np.mean(y_given_z > 0.5) 
            results['p_y_given_do_z'].append(p_positive_z)
            # ---------------------------------------------------------
            
    return results

def analyze_feature_importance(model, x_batch, y_batch, feature_names, 
                               n_interventions=5, top_k=10):
    """
    Identify most important features by measuring their impact on Y.
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    baseline_pred = model.generate(x_batch, deterministic=True)
    baseline_pred = baseline_pred.squeeze().cpu().numpy()
    
    feature_impact = []
    
    # Test each feature by setting it to 0 (or mean)
    for idx, name in enumerate(tqdm.tqdm(feature_names, desc="Analyzing features")):
        # Intervene: set feature to 0
        x_intervened = intervene_on_feature(x_batch, idx, 0.0)
        
        with torch.no_grad():
            y_pred = model.generate(x_intervened, deterministic=True)
            y_pred = y_pred.squeeze().cpu().numpy()
        
        # Measure change in predictions
        impact = np.mean(np.abs(y_pred - baseline_pred))
        feature_impact.append(impact)
    
    # Get top-k features
    feature_impact = np.array(feature_impact)
    top_indices = np.argsort(feature_impact)[-top_k:][::-1]
    
    print(f"\nTop {top_k} most influential features:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {feature_names[idx]}: impact = {feature_impact[idx]:.4f}")
    
    return top_indices, feature_impact

def plot_intervention_results(results, save_path=None):
    """
    Plot the results of intervention experiments.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    feature_name = results['feature']
    values = results['intervention_values']
    
    # Plot 1: Prediction distribution vs intervention
    ax = axes[0, 0]
    pred_means = [np.mean(pred) for pred in results['intervened_pred']]
    pred_stds = [np.std(pred) for pred in results['intervened_pred']]
    
    ax.errorbar(values, pred_means, yerr=pred_stds, fmt='o-', capsize=5)
    ax.axhline(y=np.mean(results['baseline_pred']), color='r', linestyle='--', 
               label='Baseline', alpha=0.7)
    ax.fill_between(values, 
                     np.mean(results['baseline_pred']) - np.std(results['baseline_pred']),
                     np.mean(results['baseline_pred']) + np.std(results['baseline_pred']),
                     color='r', alpha=0.1)
    ax.set_xlabel(f'Intervention Value on {feature_name}')
    ax.set_ylabel('Predicted Y (mean ± std)')
    ax.set_title(f'Effect of Intervening on {feature_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: P(Y | do(X=x))
    ax = axes[0, 1]
    ax.plot(values, results['p_y_given_do'], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel(f'Intervention Value on {feature_name}')
    ax.set_ylabel('P(Y=1 | do(X=x))')
    ax.set_title('Causal Effect: Probability of Positive Outcome')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: P(Y | do(x), z) instead of Latent space shift
    ax = axes[1, 0]
    ax.plot(values, results['p_y_given_do_z'], 'o-', color='purple', linewidth=2, markersize=8)
    ax.set_xlabel(f'Intervention Value on {feature_name}')
    ax.set_ylabel('P(Y=1 | do(X=x), Z=z)')
    ax.set_title('Direct Effect: Probability given do(x) and fixed Z')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Individual sample trajectories
    ax = axes[1, 1]
    n_samples_to_show = min(5, len(results['intervened_pred'][0]))
    
    for sample_idx in range(n_samples_to_show):
        sample_trajectory = [np.mean(pred[sample_idx]) if pred[sample_idx].ndim > 0 else pred[sample_idx] 
                             for pred in results['intervened_pred']]
        ax.plot(values, sample_trajectory, 'o-', alpha=0.6, 
                label=f'Sample {sample_idx+1} (Mean)')
    
    ax.set_xlabel(f'Intervention Value on {feature_name}')
    ax.set_ylabel('Predicted Y (Mean across targets)')
    ax.set_title('Individual Sample Responses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def counterfactual_analysis(model, x_batch, y_batch, feature_names, 
                            feature_idx, target_value, n_samples=100):
    """
    Counterfactual: What would Y be if feature had been different?
    """
    print("\n" + "="*60)
    print("COUNTERFACTUAL ANALYSIS")
    print("="*60)
    
    feature_name = feature_names[feature_idx]
    
    # Original
    with torch.no_grad():
        _, z_original = model.encode(x_batch, y_batch)
        y_original = model.generate(x_batch, deterministic=True)
        y_original = y_original.squeeze().cpu().numpy()
    
    # Counterfactual: intervene on feature
    x_counter = intervene_on_feature(x_batch, feature_idx, target_value)
    
    with torch.no_grad():
        y_counter = model.generate(x_counter, deterministic=True)
        y_counter = y_counter.squeeze().cpu().numpy()
        
        # Multiple samples for distribution
        y_samples = []
        for _ in range(n_samples):
            y_sample = model.generate(x_counter, deterministic=False)
            y_samples.append(y_sample.squeeze().cpu().numpy())
        y_samples = np.array(y_samples)
    
    # Results
    print(f"\nIntervention: Set {feature_name} to {target_value}")
    print(f"Original Y mean: {np.mean(y_original):.4f} ± {np.std(y_original):.4f}")
    print(f"Counterfactual Y mean: {np.mean(y_counter):.4f} ± {np.std(y_counter):.4f}")
    print(f"Average change: {np.mean(y_counter - y_original):.4f}")
    
    # Probability change
    # p_orig = np.mean(y_original > 0.5)
    # p_counter = np.mean(y_counter > 0.5)
    # print(f"\nP(Y=1) original: {p_orig:.4f}")
    # print(f"P(Y=1) counterfactual: {p_counter:.4f}")
    # print(f"Absolute change: {abs(p_counter - p_orig):.4f}")
    
    return {
        'original': y_original,
        'counterfactual': y_counter,
        'samples': y_samples,
        'feature': feature_name,
        'target_value': target_value
    }

def plot_counterfactual(results, save_path=None):
    """Plot counterfactual results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Original vs Counterfactual distributions
    ax = axes[0]
    ax.hist(results['original'].flatten(), bins=30, alpha=0.5, label='Original', density=True)
    ax.hist(results['counterfactual'].flatten(), bins=30, alpha=0.5, label='Counterfactual', density=True)
    ax.axvline(x=0.5, color='r', linestyle='--', label='Decision boundary')
    ax.set_xlabel('Predicted Y')
    ax.set_ylabel('Density')
    ax.set_title(f'Effect of Setting {results["feature"]} = {results["target_value"]}')
    ax.legend()
    
    # Plot 2: Individual changes
    ax = axes[1]
    n_show = min(20, len(results['original']))
    indices = np.arange(n_show)
    
    n_outputs = results['original'].shape[1] if results['original'].ndim > 1 else 1
    indices_expanded = np.repeat(indices, n_outputs)

    ax.scatter(indices_expanded, results['original'][:n_show].flatten(), label='All Originals', alpha=0.5, s=20)
    ax.scatter(indices_expanded, results['counterfactual'][:n_show].flatten(), 
               label='Counterfactual', alpha=0.7, s=50, marker='^')
    
    for i in indices:
        if n_outputs > 1:
            for j in range(n_outputs):
                ax.plot([i, i], 
                        [results['original'][i, j], results['counterfactual'][i, j]], 
                        'k-', alpha=0.1)
        else:
            ax.plot([i, i], 
                    [results['original'][i], results['counterfactual'][i]], 
                    'k-', alpha=0.3)
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Predicted Y')
    ax.set_title('Individual Sample Changes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

# ==================================================
# MAIN INTERVENTION ANALYSIS
# ==================================================

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "checkpoints/causalvae_heavy_rain_all.pt"
    
    # These should match your data
    n_features = len(FEATURE_COLS)  # Use actual number of features
    seq_length = SEQ_LENGTH   # Use actual sequence length
    
    # Feature names - USE REAL NAMES FROM FEATURE_COLS
    feature_names = FEATURE_COLS  # This is the only change needed
    
    # Intervention parameters
    INTERVENTION_VALUES = [-1, -0.5, 0, 0.5, 1]  # Standardized values
    
    print("="*60)
    print("CAUSAL INTERVENTION ANALYSIS FOR CausalVAE")
    print("="*60)
        
    # Load model and data
    model, x_batch, y_batch = load_model_and_data(
        MODEL_PATH, test_loader, n_features, seq_length
    )
    
    # === ANALYSIS 1: Feature Importance ===
    top_features, impact_scores = analyze_feature_importance(
        model, x_batch[:100], y_batch[:100], feature_names, top_k=5
    )
    
    # === ANALYSIS 2: Intervene on top features ===
    print("\n" + "="*60)
    print("INTERVENTION EXPERIMENTS")
    print("="*60)
    
    for feat_idx in top_features[:5]:  # Test top 5 features
        feat_name = feature_names[feat_idx]
        
        print(f"\n--- Intervening on {feat_name} ---")
        
        # Run intervention
        results = run_intervention_experiment(
            model, x_batch[:50], y_batch[:50],  # Use subset for speed
            feat_name, feat_idx,
            INTERVENTION_VALUES
        )
        
        # Plot results
        plot_intervention_results(
            results, 
            save_path=f"intervention_{feat_name.replace(' ', '_')}.png"
        )
    
    # === ANALYSIS 3: Counterfactual on specific feature ===
    print("\n" + "="*60)
    print("COUNTERFACTUAL ANALYSIS")
    print("="*60)
    
    # Choose a feature to analyze
    for test_feature_idx in top_features[:5]:  # Test top 5 features
        test_feature_name = feature_names[test_feature_idx]
    
        # Counterfactual: what if this feature was 2 standard deviations above mean?
        counter_results = counterfactual_analysis(
            model, x_batch[:100], y_batch[:100],
            feature_names, test_feature_idx,
            target_value=2.0  # Intervene to high value
        )
    
    plot_counterfactual(
        counter_results,
        save_path=f"images/counterfactual_{test_feature_name.replace(' ', '_')}.png"
    )
    
    print("\n" + "="*60)
    print("INTERVENTION ANALYSIS COMPLETE")
    print("="*60)