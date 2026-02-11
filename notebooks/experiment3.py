import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.fft import rfft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# 1. DATA LOADING CLASS
# ==========================================
class TimeSeriesDataset:
    def __init__(self, data_dir, labels_file, filename_col='filename', label_col='label', feature_cols=None):
        """
        Loads multivariate time series data from CSV files based on a master label file.
        """
        self.data_dir = data_dir

        # Normalize feature_cols to always be a list
        if isinstance(feature_cols, str):
            self.feature_cols = [feature_cols]
        else:
            self.feature_cols = feature_cols

        self.metadata = pd.read_csv(labels_file)

        # Verify files exist to avoid runtime errors
        self.metadata['full_path'] = self.metadata[filename_col].apply(
            lambda x: os.path.join(data_dir, str(x) if str(x).endswith('.csv') else f"{x}.csv")
        )
        # Filter out missing files
        original_count = len(self.metadata)
        self.metadata = self.metadata[self.metadata['full_path'].apply(os.path.exists)].copy()

        if len(self.metadata) < original_count:
            print(f"Warning: {original_count - len(self.metadata)} files were missing and excluded.")

        # Encode Labels (e.g., 'NoFlare' -> 0, 'Flare' -> 1)
        self.label_encoder = LabelEncoder()
        self.metadata['encoded_label'] = self.label_encoder.fit_transform(self.metadata[label_col])
        self.classes_ = self.label_encoder.classes_

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        try:
            df = pd.read_csv(row['full_path'])

            # Select specific columns (channels)
            if self.feature_cols:
                series = df[self.feature_cols].values
            else:
                series = df.values

            return series, row['encoded_label']

        except Exception as e:
            print(f"Error loading {row['full_path']}: {e}")
            return np.array([]), -1

    def get_splits(self, seed=42):
        """Loads all data and creates stratified train/test splits."""
        print(f"Loading {len(self)} time series into memory...")
        X, y = [], []

        for idx in range(len(self)):
            series, label = self[idx]
            if len(series) > 0:
                X.append(series)
                y.append(label)

        y = np.array(y)

        # Stratified Split (keeps the ratio of Flares/NoFlares consistent)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=seed
        )
        print(f"Data Split Complete: Train={len(X_train)}, Test={len(X_test)}")
        return (X_train, y_train), (X_test, y_test)

# ==========================================
# 2. FEATURE EXTRACTION ENGINE (Updated)
# ==========================================
class MultiScaleSpectralFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, channel_names, max_length=360, window_sizes=[360, 180, 90, 45], n_coeffs=5):
        """
        Extracts FFT Magnitude features from multiple sliding window scales.
        Phase info has been removed to reduce noise.
        """
        self.channel_names = channel_names
        self.max_length = max_length
        self.window_sizes = window_sizes
        self.n_coeffs = n_coeffs
        self.feature_names_ = []

    def fit(self, X, y=None):
        # Generate readable feature names during fit
        self._generate_feature_names()
        return self

    def transform(self, X):
        all_features = []

        for series in X:
            series = np.array(series)
            T, C = series.shape

            # 1. Truncate or Pad to fixed max_length
            if T > self.max_length:
                series = series[:self.max_length, :]
            elif T < self.max_length:
                padding = np.zeros((self.max_length - T, C))
                series = np.vstack([series, padding])

            sample_features = []

            # 2. Multi-Scale Windowing
            for w_size in self.window_sizes:
                n_slices = self.max_length // w_size

                for i in range(n_slices):
                    start = i * w_size
                    end = start + w_size
                    sub_window = series[start:end, :]

                    # FFT Calculation
                    fft_vals = rfft(sub_window, axis=0)

                    # Extract Magnitude (Energy)
                    # We take top 'n_coeffs' rows (Low frequency + DC)
                    mag = np.abs(fft_vals)[:self.n_coeffs, :]

                    # Flatten: Append all channels for this window slice
                    # Must match order in _generate_feature_names
                    for c_idx in range(C):
                        sample_features.extend(mag[:, c_idx])

            all_features.append(sample_features)

        return np.array(all_features)

    def _generate_feature_names(self):
        """Generates the list of string names for plotting."""
        self.feature_names_ = []
        for w_size in self.window_sizes:
            n_slices = self.max_length // w_size
            for i in range(n_slices):
                for ch_name in self.channel_names:
                    for k in range(self.n_coeffs):
                        # Construct Name: WinSize_SliceNum_Channel_FreqIndex_Type
                        self.feature_names_.append(f"Win{w_size}_Slice{i}_{ch_name}_Freq{k}_Mag")

# ==========================================
# 3. METRICS & VISUALIZATION (Updated)
# ==========================================
class SolarEvaluator:
    @staticmethod
    def calculate_tss_hss(y_true, y_pred):
        """Calculates True Skill Statistic (TSS) and Heidke Skill Score (HSS)."""
        cm = confusion_matrix(y_true, y_pred)

        # Handle binary vs multiclass (Treating Class 1 as 'Positive/Flare')
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # One-vs-Rest approach for Class 1 (Positive)
            tp = cm[1, 1]
            fn = cm[1, 0] + cm[1, 2:].sum()
            fp = cm[0, 1] + cm[2:, 1].sum()
            tn = cm.sum() - (tp + fp + fn)

        # TSS = Sensitivity + Specificity - 1
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        tss = sensitivity + specificity - 1

        # HSS
        numerator = 2 * (tp * tn - fp * fn)
        denominator = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
        hss = numerator / denominator if denominator != 0 else 0

        return tss, hss

    @staticmethod
    def find_best_threshold(pipeline, X, y):
        """
        Scans probabilities from 0.1 to 0.9 to find the threshold that
        maximizes the TSS score on the provided dataset.
        """
        probs = pipeline.predict_proba(X)[:, 1] # Probability of Class 1 (Flare)

        thresholds = np.arange(0.1, 0.9, 0.05)
        best_tss = -1
        best_th = 0.5

        print("\nScanning thresholds for optimal TSS...")
        for th in thresholds:
            preds = (probs >= th).astype(int)
            tss, hss = SolarEvaluator.calculate_tss_hss(y, preds)

            if tss > best_tss:
                best_tss = tss
                best_th = th

        print(f" -> Best Threshold Found: {best_th:.2f} (Max TSS: {best_tss:.4f})")
        return best_th

    @staticmethod
    def plot_top_features(importances, feature_names, top_n=20):
        """Plots the most important features extracted by the model."""
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        df = df.sort_values(by='Importance', ascending=False).head(top_n)

        plt.figure(figsize=(12, 10))
        sns.barplot(data=df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
        plt.title(f"Top {top_n} Most Important Spectral Features", fontsize=16)
        plt.xlabel("Gini Importance Score", fontsize=12)
        plt.ylabel("Feature Name", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

# ==========================================
# 4. MAIN EXECUTION FLOW
# ==========================================

# A. Load Data
# ---------------------------------------------------------
# UPDATE THIS PATH TO MATCH YOUR LOCAL FOLDER
data_root = 'D:/workspace/spectral_analysis/data/'
# ---------------------------------------------------------

# Explicitly define channel names (used for naming features later)
target_channels = ['p3_flux_ic', 'p5_flux_ic', 'p7_flux_ic']

dataset = TimeSeriesDataset(
    data_dir=data_root+'raw/',
    labels_file=data_root+'SEP_class_labels.csv',
    filename_col='File',
    label_col='Label',
    feature_cols=target_channels
)

(X_train, y_train), (X_test, y_test) = dataset.get_splits()

# B. Initialize Featurizer
# We focus on Magnitude only, across 4 window scales
featurizer = MultiScaleSpectralFeaturizer(
    channel_names=target_channels,
    max_length=360,
    window_sizes=[360, 180, 90, 45],
    n_coeffs=5
)

# C. Build Pipeline with Class Balancing
rf_model = RandomForestClassifier(
    n_estimators=200,             # More trees for stability
    class_weight='balanced_subsample', # FIX: Handes Class Imbalance automatically
    min_samples_leaf=4,           # Regularization: Prevents overfitting to noise
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ('spec', featurizer),
    ('rf', rf_model)
])

# D. Train
print("\nTraining Multi-Scale Spectral Model...")
pipeline.fit(X_train, y_train)

# E. Threshold Tuning (The "Solar Physics" Optimization)
# We tune the threshold on X_train to maximize TSS, then apply it to X_test.
# (Ideally, use a separate Validation set here if you have enough data).
best_threshold = SolarEvaluator.find_best_threshold(pipeline, X_train, y_train)

# F. Final Evaluation on Test Set
print(f"\nEvaluating on Test Set using Threshold {best_threshold:.2f}...")

# Get Probabilities
y_probs_test = pipeline.predict_proba(X_test)[:, 1]
# Apply Optimized Threshold
y_pred_optimized = (y_probs_test >= best_threshold).astype(int)

# 1. Standard Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_optimized))

# 2. Solar Metrics
tss, hss = SolarEvaluator.calculate_tss_hss(y_test, y_pred_optimized)
print("="*40)
print(f"SOLAR PHYSICS METRICS (Optimized):")
print(f"TSS (True Skill Statistic): {tss:.4f}")
print(f"HSS (Heidke Skill Score):   {hss:.4f}")
print("="*40)

# G. Visualize Feature Importance
trained_featurizer = pipeline.named_steps['spec']
feature_names = trained_featurizer.feature_names_
importances = pipeline.named_steps['rf'].feature_importances_

print(f"Total Features Generated: {len(feature_names)}")
SolarEvaluator.plot_top_features(importances, feature_names, top_n=20)