import r2_download as hd
import pandas as pd
import numpy as np
import tqdm


import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import timedelta
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    confusion_matrix, classification_report, brier_score_loss, f1_score
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

import netCDF4

# === Reproducibility ===
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = hd._default_data_dir()

# === Station files (4 inland stations, CSV from R2) ===
# After R2 download, files land at {DATA_DIR}/precipitation-nowcasting/weather_stations/...
PRECIP_DIR = f"{DATA_DIR}/precipitation-nowcasting"

STATION_FILES = {
    'cer': 'weather_stations/CER_consolid_f15.csv',
    'jun': 'weather_stations/JUN_consolid_f15.csv',
    'merc': 'weather_stations/MERC_consolid_f15.csv',
    'mira': 'weather_stations/MIRA_consolid_f15.csv',
}

# === Column mapping: harmonized name -> list of candidate column names ===
# Multi-candidate lookup handles naming inconsistencies between stations
COLUMN_MAP = {
    'rain_mm': ['Rain_mm_Tot'],
    'temp_c': ['AirTC_Avg'],
    'rh_avg': ['RH_Avg'],
    'rh_max': ['RH_Max'],
    'rh_min': ['RH_Min'],
    'solar_kw': ['SlrkW_Avg'],
    'net_rad_wm2': ['NR_Wm2_Avg'],
    'wind_speed_ms': ['WS_ms_Avg'],
    'wind_dir': ['WindDir'],
    'soil_moisture_1': ['VW_Avg', 'VW'],
    'soil_moisture_2': ['VW_2_Avg', 'VW_2'],
    'soil_moisture_3': ['VW_3_Avg', 'VW_3'],
    'leaf_wetness': ['LWmV_Avg'],
    'leaf_wet_minutes': ['LWMWet_Tot'],
}


def load_station(name, filename):
    """Load a station CSV, harmonize columns, set datetime index."""
    path = f"{PRECIP_DIR}/{filename}"
    print(f"  Loading {name} from {filename}...", end=" ")
    df = pd.read_csv(path, low_memory=False)
    print(f"({df.shape[0]:,} rows, {df.shape[1]} cols)")

    # === Parse timestamp (M/D/YYYY H:MM format) ===
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%m/%d/%Y %H:%M')
    df = df.set_index('TIMESTAMP').sort_index()

    # === Multi-candidate column lookup ===
    rename = {}
    for harmonized, candidates in COLUMN_MAP.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename[candidate] = harmonized
                break  # use the first match

    df = df[list(rename.keys())].rename(columns=rename)

    # === Force numeric (some columns loaded as object due to mixed types) ===
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.attrs['station'] = name
    return df


# === Load all stations ===
print("Loading 4 inland weather stations...")
stations = {}
for name, fname in tqdm(STATION_FILES.items(), desc="Stations"):
    stations[name] = load_station(name, fname)

print(f"\nLoaded {len(stations)} stations.")
for name, df in stations.items():
    print(f"  {name:8s}: {df.index.min().date()} -> {df.index.max().date()}"
          f"  ({df.shape[0]:>8,} rows, {len(df.columns)} cols)")
    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        print(f"           Missing variables: {missing}")

# === Target station for baseline model ===
TARGET_STATION = 'jun'  # El Junco — highest-elevation, most rainfall
STATIONS = list(STATION_FILES.keys())

# === Merge all stations into wide DataFrame ===
# Prefix each station's columns with its abbreviation, then concat on shared time index
print("Merging stations into multi-station DataFrame...")
prefixed = []
for name, stn_df in stations.items():
    renamed = stn_df.add_prefix(f'{name}_')
    prefixed.append(renamed)

df = pd.concat(prefixed, axis=1, sort=True)

# === Drop columns that are entirely NaN (e.g., cer_net_rad_wm2 — CER lacks this sensor) ===
all_nan_cols = [c for c in df.columns if df[c].isnull().all()]
if all_nan_cols:
    print(f"  Dropping {len(all_nan_cols)} entirely-NaN columns: {all_nan_cols}")
    df = df.drop(columns=all_nan_cols)

print(f"Wide DataFrame: {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"  Time range: {df.index.min()} -> {df.index.max()}")
print(f"\nMissing data before imputation (showing > 0% only):")
missing = df.isnull().mean() * 100
print(missing[missing > 0].round(1).to_string())


# === Per-station imputation on the wide DataFrame ===
for stn in STATIONS:
    # Temperature, humidity: interpolate up to 6h (24 steps at 15-min)
    for var in ['temp_c', 'rh_avg', 'rh_max', 'rh_min']:
        col = f'{stn}_{var}'
        if col in df.columns:
            df[col] = df[col].interpolate(method='time', limit=24)

    # Solar / net radiation: interpolate up to 6h
    for var in ['solar_kw', 'net_rad_wm2']:
        col = f'{stn}_{var}'
        if col in df.columns:
            df[col] = df[col].interpolate(method='time', limit=24)

    # Soil moisture: interpolate up to 6h (physically smooth)
    for var in ['soil_moisture_1', 'soil_moisture_2', 'soil_moisture_3']:
        col = f'{stn}_{var}'
        if col in df.columns:
            df[col] = df[col].interpolate(method='time', limit=24)

    # Wind speed: forward-fill short gaps, then interpolate up to 2h
    col = f'{stn}_wind_speed_ms'
    if col in df.columns:
        df[col] = df[col].ffill(limit=4)
        df[col] = df[col].interpolate(method='time', limit=8)

    # Wind direction: forward-fill only (circular — never interpolate)
    col = f'{stn}_wind_dir'
    if col in df.columns:
        df[col] = df[col].ffill(limit=8)

    # Leaf wetness: forward-fill (sensor state change)
    for var in ['leaf_wetness', 'leaf_wet_minutes']:
        col = f'{stn}_{var}'
        if col in df.columns:
            df[col] = df[col].ffill(limit=8)

    # Precipitation: zero-fill + binary rain indicator
    rain_col = f'{stn}_rain_mm'
    if rain_col in df.columns:
        df[f'{stn}_rain_missing'] = df[rain_col].isnull().astype(float)
        df[rain_col] = df[rain_col].fillna(0.0)

# === Global forward/backward fill for remaining short gaps ===
df = df.ffill(limit=96).bfill(limit=96)

# === Fill any remaining NaN with 0 (long sensor outages) ===
still_nan = df.isnull().sum().sum()
if still_nan > 0:
    n_cols_with_nan = (df.isnull().sum() > 0).sum()
    print(f"Filling {still_nan:,} remaining NaN across {n_cols_with_nan} columns with 0 "
          "(long sensor outages)")
df = df.fillna(0.0)

# === Defragment DataFrame (avoids PerformanceWarning in feature engineering) ===
df = df.copy()

print(f"Final DataFrame: {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"Missing data after imputation: {df.isnull().sum().sum()}")

# === Cyclical time features (shared across stations) ===
df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

# === Season one-hot encoding ===
# Define season based on month: Winter (Dec-Feb), Spring (Mar-May), Summer (Jun-Aug), Autumn (Sep-Nov)
month = df.index.month
df['season_winter'] = ((month >= 12) | (month <= 2)).astype(int)
df['season_spring'] = ((month >= 3) & (month <= 5)).astype(int)
df['season_summer'] = ((month >= 6) & (month <= 8)).astype(int)
df['season_autumn'] = ((month >= 9) & (month <= 11)).astype(int)

# === Per-station derived features ===
for stn in STATIONS:
    # Wind vector decomposition: circular direction -> linear Wx/Wy
    wd_col = f'{stn}_wind_dir'
    ws_col = f'{stn}_wind_speed_ms'
    if wd_col in df.columns and ws_col in df.columns:
        wd_rad = np.deg2rad(df[wd_col])
        df[f'{stn}_wind_x'] = df[ws_col] * np.cos(wd_rad)
        df[f'{stn}_wind_y'] = df[ws_col] * np.sin(wd_rad)

    # Dewpoint approximation (Magnus formula)
    # Td = (237.3 * alpha) / (17.27 - alpha)
    # alpha = (17.27 * T) / (237.3 + T) + ln(RH / 100)
    temp_col = f'{stn}_temp_c'
    rh_col = f'{stn}_rh_avg'
    if temp_col in df.columns and rh_col in df.columns:
        T = df[temp_col]
        RH = df[rh_col].clip(lower=1)  # avoid log(0)
        alpha = (17.27 * T) / (237.3 + T) + np.log(RH / 100)
        df[f'{stn}_dewpoint'] = (237.3 * alpha) / (17.27 - alpha)
        df[f'{stn}_dewpoint_depression'] = T - df[f'{stn}_dewpoint']

    # Soil moisture tendency (3h change) — signals subsurface water movement
    sm_col = f'{stn}_soil_moisture_1'
    if sm_col in df.columns:
        df[f'{stn}_soil_moist_tend_3h'] = df[sm_col].diff(periods=12)

    # Rolling statistics: accumulated rain, temp/wind/humidity trends
    for window, wlabel in [(4, '1h'), (12, '3h'), (24, '6h')]:
        rain_col = f'{stn}_rain_mm'
        if rain_col in df.columns:
            df[f'{stn}_rain_sum_{wlabel}'] = df[rain_col].rolling(window, min_periods=1).sum()

        if temp_col in df.columns:
            df[f'{stn}_temp_mean_{wlabel}'] = df[temp_col].rolling(window, min_periods=1).mean()
            df[f'{stn}_temp_std_{wlabel}'] = df[temp_col].rolling(window, min_periods=1).std()

        if ws_col in df.columns:
            df[f'{stn}_wind_mean_{wlabel}'] = df[ws_col].rolling(window, min_periods=1).mean()

        if rh_col in df.columns:
            df[f'{stn}_rh_mean_{wlabel}'] = df[rh_col].rolling(window, min_periods=1).mean()

print(f"Features after engineering: {len(df.columns)} columns")
print(f"  Shared time features: 4")
print(f"  Season features: 4")
print(f"  Per-station columns: ~{(len(df.columns) - 8) // len(STATIONS)} each")

# 4.3 Define Extreme Event Labels
# === Precipitation labels (per-station) ===
HORIZONS = {'3h': 12, '6h': 24, '12h': 48}  # steps at 15-min resolution

# Forward-looking accumulated rainfall for each station and horizon
for stn in STATIONS:
    rain_col = f'{stn}_rain_mm'
    for label, steps in HORIZONS.items():
        df[f'rain_future_{label}_{stn}'] = (
            df[rain_col]
            .rolling(steps, min_periods=1)
            .sum()
            .shift(-steps)
        )

# === Compute thresholds from training period (target station, to avoid leakage) ===
train_mask = df.index < '2023-01-01'
print(f"Precipitation accumulation statistics ({TARGET_STATION}, training period, wet > 0):")

thresholds = {}
for label, steps in HORIZONS.items():
    col = f'rain_future_{label}_{TARGET_STATION}'
    wet = df.loc[train_mask, col]
    wet = wet[wet > 0]
    p95 = wet.quantile(0.95)
    p99 = wet.quantile(0.99)
    thresholds[label] = max(p95, 2.0)
    print(f"  {label}: mean={wet.mean():.2f}mm, p50={wet.median():.2f}mm, "
          f"p95={p95:.2f}mm, p99={p99:.2f}mm -> threshold={thresholds[label]:.2f}mm")

# === Create binary labels for all stations ===
for stn in STATIONS:
    for label in HORIZONS:
        col = f'rain_future_{label}_{stn}'
        thresh = thresholds[label]
        df[f'heavy_rain_{label}_{stn}'] = (df[col] >= thresh).astype(float)
        df.loc[df[col].isnull(), f'heavy_rain_{label}_{stn}'] = np.nan

# === Default targets (from TARGET_STATION) for the baseline model ===
for label in HORIZONS:
    df[f'heavy_rain_{label}'] = df[f'heavy_rain_{label}_{TARGET_STATION}']

# === Temperature anomaly (target station) ===
temp_col = f'{TARGET_STATION}_temp_c'
df['day_of_year'] = df.index.dayofyear
daily_clim = df.loc[train_mask].groupby('day_of_year')[temp_col].agg(['mean', 'std'])
daily_clim['mean_smooth'] = daily_clim['mean'].rolling(15, center=True, min_periods=5).mean()
daily_clim['std_smooth'] = daily_clim['std'].rolling(15, center=True, min_periods=5).mean()
daily_clim = daily_clim.bfill().ffill()

df['temp_clim_mean'] = df['day_of_year'].map(daily_clim['mean_smooth'])
df['temp_clim_std'] = df['day_of_year'].map(daily_clim['std_smooth'])
df['temp_anomaly'] = (df[temp_col] - df['temp_clim_mean']) / df['temp_clim_std']
df['temp_extreme'] = (df['temp_anomaly'].abs() > 2).astype(float)

print(f"\nLabel balance (full dataset, target station: {TARGET_STATION}):")
for label in HORIZONS:
    col = f'heavy_rain_{label}'
    pos_rate = df[col].mean()
    n_pos = df[col].sum()
    print(f"  heavy_rain_{label}: {pos_rate:.3%} positive ({n_pos:.0f} events)")
print(f"  temp_extreme: {df['temp_extreme'].mean():.3%} positive ({df['temp_extreme'].sum():.0f} events)")

# === Show per-station label comparison ===
print(f"\nPer-station 3h heavy rain rates:")
for stn in STATIONS:
    col = f'heavy_rain_3h_{stn}'
    rate = df[col].mean()
    n = df[col].sum()
    print(f"  {stn}: {rate:.3%} ({n:.0f} events)")

### 4.4 Train / Validation / Test Split
# === Define split boundaries ===
TRAIN_END = pd.Timestamp('2023-01-01')
VAL_END = pd.Timestamp('2024-07-01')
EMBARGO_STEPS = 48  # 12 hours at 15-min resolution

# === Drop temporary columns, keep features + labels ===
drop_cols = [c for c in df.columns if c.startswith('rain_future_')]
drop_cols += ['day_of_year', 'temp_clim_mean', 'temp_clim_std']
df = df.drop(columns=drop_cols)

# === Split by timestamp ===
train_df = df[df.index < TRAIN_END].copy()
# Embargo: skip 12h after train boundary
val_start = TRAIN_END + timedelta(hours=12)
val_df = df[(df.index >= val_start) & (df.index < VAL_END)].copy()
# Embargo: skip 12h after val boundary
test_start = VAL_END + timedelta(hours=12)
test_df = df[df.index >= test_start].copy()

print(f"Train: {train_df.index.min().date()} → {train_df.index.max().date()} ({len(train_df):,} rows)")
print(f"Val:   {val_df.index.min().date()} → {val_df.index.max().date()} ({len(val_df):,} rows)")
print(f"Test:  {test_df.index.min().date()} → {test_df.index.max().date()} ({len(test_df):,} rows)")

# === Check label balance in each split ===
for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    for label in HORIZONS:
        col = f'heavy_rain_{label}'
        n_pos = split_df[col].sum()
        rate = split_df[col].mean()
        print(f"  {split_name} heavy_rain_{label}: {n_pos:.0f} events ({rate:.2%})")


# === Define feature columns (everything except labels and intermediates) ===
LABEL_COLS = [f'heavy_rain_{h}' for h in HORIZONS]
for stn in STATIONS:
    LABEL_COLS += [f'heavy_rain_{h}_{stn}' for h in HORIZONS]
LABEL_COLS += ['temp_extreme', 'temp_anomaly']

FEATURE_COLS = [c for c in df.columns if c not in LABEL_COLS]

print(f"Feature columns ({len(FEATURE_COLS)}):")
for i, c in enumerate(FEATURE_COLS[:15]):
    print(f"  {i:3d}. {c}")
if len(FEATURE_COLS) > 15:
    print(f"  ... and {len(FEATURE_COLS) - 15} more")

# === Compute train statistics ===
train_mean = train_df[FEATURE_COLS].mean()
train_std = train_df[FEATURE_COLS].std()
# Avoid division by zero for constant columns
train_std = train_std.replace(0, 1)

# === Normalize all splits ===
for split_df in [train_df, val_df, test_df]:
    split_df[FEATURE_COLS] = (split_df[FEATURE_COLS] - train_mean) / train_std

print(f"\nNormalization complete (fitted on training data only).")

os.makedirs(f"{PRECIP_DIR}/data_processed", exist_ok=True)

train_df.to_csv(f"{PRECIP_DIR}/data_processed/train_data.csv")
val_df.to_csv(f"{PRECIP_DIR}/data_processed/val_data.csv") 
test_df.to_csv(f"{PRECIP_DIR}/data_processed/test_data.csv")