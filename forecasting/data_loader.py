# forecasting/data_loader.py

from pyexpat import features

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class LSTMDataLoader:
    """
    Prepares merged_lstm_core.csv for LSTM training.

    Input features (what LSTM sees):
        1. total_tokens_15min   - token volume per 15 min
        2. requests_per_15min   - number of LLM requests
        3. avg_gpu_power_w      - GPU power draw in watts
        4. hour_sin             - time of day sine encoding
        5. hour_cos             - time of day cosine encoding
        6. DoW                  - day of week (0=Mon, 6=Sun)

    Target (what LSTM predicts):
        TLHC - thermal load heat coefficient (already normalized)
    """

    # FEATURE_COLS = [
    #     'total_tokens_15min',
    #     'requests_per_15min',
    #     'avg_gpu_power_w',
    #     'hour_sin',
    #     'hour_cos',
    #     'DoW',
    # ]##
    FEATURE_COLS = [
        'T_out-0',
        'T_out-1',
        'T_out-2',
        'T_out-3',
        'T_celCC-0',
        'T_celCC-1',
        'T_celCC-2',
        'T_celCC-3',
        'hour_sin',
        'hour_cos',
    ]
    TARGET_COL = 'TLHC'

    def __init__(self, csv_path: str, lookback: int = 8):
        """
        csv_path : path to merged_lstm_core.csv
        lookback : how many past steps to use as input
                   8 steps x 15 min = 2 hours of history
        """
        self.lookback    = lookback
        self.df          = self._load(csv_path)
        self.feature_min = None
        self.feature_max = None

    def _load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.dropna(
            subset=self.FEATURE_COLS + [self.TARGET_COL]
        )
        print(f"Loaded      : {len(df)} rows")
        print(f"Period      : {df['timestamp'].iloc[0]}"
              f" to {df['timestamp'].iloc[-1]}")
        return df

    def _normalize(self) -> np.ndarray:
        """
        Z-score standardization: (x - mean) / std
        This matches the normalization used for TLHC target
        and gives better LSTM training stability.
        """
        features          = self.df[self.FEATURE_COLS].copy()
        self.feature_mean = features.mean()
        self.feature_std  = features.std()

        normalized = (
            (features - self.feature_mean) /
            (self.feature_std + 1e-8)
        )
        return normalized.values.astype(np.float32)

    def _build_sequences(
        self,
        features: np.ndarray,
        targets:  np.ndarray
    ):
        """
        Builds sliding window sequences.

        For each position i starting from lookback:
            X[i] = features from (i - lookback) to i
            y[i] = target at position i

        This teaches the LSTM:
            given last 2 hours of data -> predict next heat
        """
        X, y = [], []
        for i in range(self.lookback, len(features)):
            X.append(features[i - self.lookback : i])
            y.append(targets[i])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        print(f"Sequences   : {len(X)}")
        print(f"Input shape : {X.shape}"
              f"  (sequences, lookback, features)")
        print(f"Target shape: {y.shape}")
        return X, y

    def get_loaders(
        self,
        batch_size: int   = 32,
        val_split:  float = 0.2,
    ):
        """
        Returns PyTorch DataLoaders for training and validation.

        80% of data goes to training.
        20% goes to validation.
        shuffle=False keeps time order intact.
        """
        features = self._normalize()
        targets  = self.df[self.TARGET_COL].values.astype(
            np.float32
        )

        X, y = self._build_sequences(features, targets)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=val_split,
            shuffle=True,
            random_state=42
        )

        print(f"Train size  : {len(X_train)} sequences")
        print(f"Val size    : {len(X_val)} sequences")

        train_loader = DataLoader(
            _ThermalDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            _ThermalDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
        )
        return train_loader, val_loader

    def get_full_sequence(self):
        """
        Returns full X and y without splitting.
        Used for evaluation and plotting predictions.
        """
        features = self._normalize()
        targets  = self.df[self.TARGET_COL].values.astype(
            np.float32
        )
        return self._build_sequences(features, targets)


class _ThermalDataset(Dataset):
    """
    Simple PyTorch Dataset wrapper.
    Holds input output pairs for the DataLoader.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]