"""
Custom feature transformers for TDE anomaly detection.
This module contains shared transformers that can be used by both training and testing scripts.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureWeightScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer that applies feature-specific weights before scaling.
    This allows us to emphasize R² features (60% weight) over other features (40% weight).
    
    Used in the anomaly detection pipeline to prioritize R² (r_squared_t53) features
    which are key indicators of TDE light curve signatures.
    """
    def __init__(self, feature_weights=None, base_weight=0.67):
        """
        Args:
            feature_weights: Dict mapping feature names to weights (e.g., {'r_squared_t53': 1.5})
            base_weight: Base weight for features not in feature_weights (default: 0.67 for 40/60 split)
        """
        self.feature_weights = feature_weights if feature_weights is not None else {}
        self.base_weight = base_weight
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        """Store feature names for weighting."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            # If X is a numpy array, create generic feature names
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        return self
    
    def transform(self, X):
        """Apply feature weights to X."""
        if isinstance(X, pd.DataFrame):
            # Apply weights to DataFrame
            X_weighted = X.copy()
            for feature_name in X.columns:
                if feature_name in self.feature_names_:
                    weight = self.feature_weights.get(feature_name, self.base_weight)
                    X_weighted[feature_name] = X[feature_name] * weight
            return X_weighted
        else:
            # Apply weights to numpy array
            X_weighted = X.copy()
            for i, feature_name in enumerate(self.feature_names_):
                if i < X.shape[1]:
                    weight = self.feature_weights.get(feature_name, self.base_weight)
                    X_weighted[:, i] = X[:, i] * weight
            return X_weighted
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for sklearn compatibility."""
        if input_features is not None:
            return np.array(input_features)
        if self.feature_names_ is not None:
            return np.array(self.feature_names_)
        return None

