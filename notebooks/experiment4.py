### Both Magnitude and Phase Features ###

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
        self.data_dir = data_dir

        # Store original feature names for later use
        if isinstance(feature_cols, str):
            self.feature_cols = [feature_cols]
        else:
            self.feature_cols = feature_cols

        self.metadata = pd.read_csv(labels_file)

        # Verify files exist
        self.metadata['full_path'] = self.metadata[filename_col].apply(
            lambda x: os.path.join(data_dir, str(x) if str(x).endswith('.csv') else f"{x}.csv")
        )
        self.metadata = self.metadata[self.metadata['full_path'].apply(os.path.exists)].copy()

        # Encode Labels
        self.label_encoder = LabelEncoder()
        self.metadata['encoded_label'] = self.label_encoder.fit_transform(self.metadata[label_col])
        self.classes_ = self.label_encoder.classes_

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        try:
            df = pd.read_csv(row['full_path'])
            if self.feature_cols:
                series = df[self.feature_cols].values
            else:
                series = df.values
            return series, row['encoded_label']
        except Exception as e:
            print(f"Error loading {row['full_path']}: {e}")
            return np.array([]), -1

    def get_splits(self, seed=42):
        print(f"Loading {len(self)} time series...")
        X, y = [], []
        for idx in range(len(self)):
            series, label = self[idx]
            if len(series) > 0:
                X.append(series)
                y.append(label)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        return (X_train, y_train), (X_test, y_test)

# ==========================================
# 2. FEATURE EXTRACTION ENGINE (OOPS)
# ==========================================
class MultiScaleSpectralFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, channel_names, max_length=360, window_sizes=[360, 180, 90, 45], n_coeffs=5):
        """
        Args:
            channel_names (list): List of strings (e.g. ['p3_flux', 'p5_flux'])
            max_length (int): Total length to truncate input to (e.g., 360 mins).
            window_sizes (list): List of window sizes to slice the 360 mins into.
            n_coeffs (int): Number of FFT coefficients to keep per window/channel.
        """
        self.channel_names = channel_names
        self.max_length = max_length
        self.window_sizes = window_sizes
        self.n_coeffs = n_coeffs
        self.feature_names_ = [] # Stores descriptive names

    def fit(self, X, y=None):
        # Generate feature names using the real channel names
        self._generate_feature_names()
        return self

    def transform(self, X):
        all_features = []

        for series in X:
            # 1. Truncate/Pad to fixed max_length (360)
            series = np.array(series)
            T, C = series.shape

            if T > self.max_length:
                series = series[:self.max_length, :]
            elif T < self.max_length:
                padding = np.zeros((self.max_length - T, C))
                series = np.vstack([series, padding])

            # 2. Extract Features for each Window Size
            sample_features = []

            for w_size in self.window_sizes:
                # Calculate how many slices fit in max_length
                n_slices = self.max_length // w_size

                for i in range(n_slices):
                    start = i * w_size
                    end = start + w_size
                    sub_window = series[start:end, :] # Shape (w_size, Channels)

                    # FFT
                    fft_vals = rfft(sub_window, axis=0) # Shape (freqs, Channels)

                    # Extract Mag and Phase
                    mag = np.abs(fft_vals)[:self.n_coeffs, :]   # Top k
                    phase = np.angle(fft_vals)[:self.n_coeffs, :] # Top k

                    # Flatten and add to vector
                    # Note: We must flatten in the EXACT same order as _generate_feature_names
                    # Order loop: Channel -> Coeff
                    for c_idx in range(C):
                        sample_features.extend(mag[:, c_idx])   # All mags for this channel
                        sample_features.extend(phase[:, c_idx]) # All phases for this channel

            all_features.append(sample_features)

        return np.array(all_features)

    def _generate_feature_names(self):
        """Generates readable names for every single feature column."""
        self.feature_names_ = []

        for w_size in self.window_sizes:
            n_slices = self.max_length // w_size
            for i in range(n_slices):
                # Using Real Channel Names instead of Index
                for ch_name in self.channel_names:
                    # For each coeff (Magnitude)
                    for k in range(self.n_coeffs):
                        self.feature_names_.append(f"Win{w_size}_Slice{i}_{ch_name}_Freq{k}_Mag")
                    # For each coeff (Phase)
                    for k in range(self.n_coeffs):
                        self.feature_names_.append(f"Win{w_size}_Slice{i}_{ch_name}_Freq{k}_Phase")

# ==========================================
# 3. METRICS & VISUALIZATION
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
data_root = 'D:/workspace/spectral_analysis/data/'
# Explicitly define channel names here
target_channels = ['p3_flux_ic', 'p5_flux_ic', 'p7_flux_ic']

dataset = TimeSeriesDataset(
    data_dir=data_root+'raw/',
    labels_file=data_root+'SEP_class_labels.csv',
    filename_col='File',
    label_col='Label',
    feature_cols=target_channels
)

(X_train, y_train), (X_test, y_test) = dataset.get_splits()

# B. Initialize Featurizer with REAL channel names
featurizer = MultiScaleSpectralFeaturizer(
    channel_names=target_channels, # <--- Passing names here
    max_length=360,
    window_sizes=[360, 180, 90, 45],
    n_coeffs=5
)

# C. Build Pipeline
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
pipeline = Pipeline([
    ('spec', featurizer),
    ('rf', rf_model)
])

# D. Train
print("\nTraining Multi-Scale Spectral Model...")
pipeline.fit(X_train, y_train)

# E. Evaluate (Standard + Solar Metrics)
print("\nEvaluating on Test Set...")
y_pred = pipeline.predict(X_test)

# Print Standard Report
print(classification_report(y_test, y_pred))

# Calculate and Print Solar Metrics
tss, hss = SolarEvaluator.calculate_tss_hss(y_test, y_pred)
print("="*30)
print(f"Solar Physics Metrics:")
print(f"TSS (True Skill Statistic): {tss:.4f}")
print(f"HSS (Heidke Skill Score):   {hss:.4f}")
print("="*30)

# F. Feature Importance Analysis
trained_featurizer = pipeline.named_steps['spec']
feature_names = trained_featurizer.feature_names_
importances = pipeline.named_steps['rf'].feature_importances_

print(f"Total Features Generated: {len(feature_names)}")
SolarEvaluator.plot_top_features(importances, feature_names, top_n=25)
