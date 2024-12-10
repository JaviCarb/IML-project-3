from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
from scipy.stats import skew, kurtosis

############################################
# MissingHandler Transformer
############################################

class MissingHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method='drop'):
        """
        method : 'drop', 'indicator', or 'impute'
        """
        self.method = method
        # Sensor indexing
        self.hands_indices = np.arange(1, 11)   # sensors 3–12
        self.chest_indices = np.arange(11, 21)  # sensors 13–22
        self.feet_indices = np.arange(21, 31)   # sensors 23–32

    def fit(self, X, y=None):
        n_samples, n_sensors, n_timepoints = X.shape
        missing_mask = np.all(np.isnan(X), axis=2)

        if self.method in ['indicator', 'impute']:
            self.median_profiles_ = np.full((n_sensors, n_timepoints), np.nan)
            for s in range(n_sensors):
                present_samples = ~missing_mask[:, s]
                if np.any(present_samples):
                    sensor_data = X[present_samples, s, :]
                    median_profile = np.nanmedian(sensor_data, axis=0)
                else:
                    # If no present samples for a sensor
                    median_profile = np.zeros(n_timepoints)
                self.median_profiles_[s, :] = median_profile
        return self

    def transform(self, X):
        n_samples, n_sensors, n_timepoints = X.shape
        missing_mask = np.all(np.isnan(X), axis=2)

        if self.method == 'drop':
            keep_mask = ~np.any(missing_mask, axis=1)
            return X[keep_mask, :, :]

        elif self.method == 'indicator':
            hands_missing = np.any(missing_mask[:, self.hands_indices], axis=1).astype(float)
            chest_missing = np.any(missing_mask[:, self.chest_indices], axis=1).astype(float)
            feet_missing = np.any(missing_mask[:, self.feet_indices], axis=1).astype(float)

            X_imputed = X.copy()
            for i in range(n_samples):
                for s in range(n_sensors):
                    if missing_mask[i, s]:
                        X_imputed[i, s, :] = self.median_profiles_[s, :]

            # Add three new sensors for indicators
            hands_indicator_array = np.tile(hands_missing[:, np.newaxis, np.newaxis], (1, 1, n_timepoints))
            chest_indicator_array = np.tile(chest_missing[:, np.newaxis, np.newaxis], (1, 1, n_timepoints))
            feet_indicator_array = np.tile(feet_missing[:, np.newaxis, np.newaxis], (1, 1, n_timepoints))

            X_final = np.concatenate([X_imputed, hands_indicator_array, chest_indicator_array, feet_indicator_array], axis=1)
            return X_final

        elif self.method == 'impute':
            X_imputed = X.copy()
            missing_mask = np.all(np.isnan(X), axis=2)
            for i in range(n_samples):
                for s in range(n_sensors):
                    if missing_mask[i, s]:
                        X_imputed[i, s, :] = self.median_profiles_[s, :]
            return X_imputed

        else:
            return X




############################################
# FeatureExtractor Transformer
############################################

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, segment=False, n_segments=4, feature_level='basic'):
        self.segment = segment
        self.n_segments = n_segments
        self.feature_level = feature_level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X
        if self.segment:
            data = self._segment_time_series(data, self.n_segments)
        features = self._extract_features(data, self.feature_level)
        return features

    def _segment_time_series(self, data, n_segments):
        n_samples, n_sensors, n_t = data.shape
        if n_t % n_segments != 0:
            raise ValueError("Number of timesteps not divisible by n_segments.")
        seg_length = n_t // n_segments
        return data.reshape(n_samples, n_sensors, n_segments, seg_length)

    def _extract_features(self, data, feature_level):
        if data.ndim == 3:
            n_samples, n_sensors, n_t = data.shape
            n_segments = 1
            data = data[:, :, np.newaxis, :]
        else:
            n_samples, n_sensors, n_segments, seg_length = data.shape

        def iqr(x):
            return np.nanpercentile(x, 75) - np.nanpercentile(x, 25)

        all_features = []
        for i in range(n_samples):
            sample_feats = []
            for s in range(n_sensors):
                for seg in range(n_segments):
                    segment_data = data[i, s, seg, :]
                    mean_val = np.nanmean(segment_data)
                    median_val = np.nanmedian(segment_data)
                    std_val = np.nanstd(segment_data)
                    var_val = np.nanvar(segment_data)
                    iqr_val = iqr(segment_data)
                    min_val = np.nanmin(segment_data)
                    max_val = np.nanmax(segment_data)
                    range_val = max_val - min_val
                    if std_val < 1e-12:
                        skew_val = 0.0
                        kurt_val = 0.0
                    else:
                        skew_val = skew(segment_data, nan_policy='omit')
                        kurt_val = kurtosis(segment_data, nan_policy='omit')

                    seg_feats = [mean_val, median_val, std_val, var_val, iqr_val, 
                                 min_val, max_val, range_val, skew_val, kurt_val]

                    if feature_level == 'extended':
                        fft_vals = np.fft.rfft(segment_data)
                        fft_mag = np.abs(fft_vals)
                        freqs = np.fft.rfftfreq(segment_data.size, d=1.0)

                        dom_freq_idx = np.nanargmax(fft_mag) if fft_mag.size > 0 else 0
                        dom_freq_val = dom_freq_idx
                        if np.sum(fft_mag) > 0:
                            spectral_centroid = np.sum(freqs * fft_mag) / np.sum(fft_mag)
                        else:
                            spectral_centroid = 0.0

                        max_freq = freqs[-1] if freqs.size > 0 else 1.0
                        low_mask = freqs < (0.2 * max_freq)
                        mid_mask = (freqs >= (0.2 * max_freq)) & (freqs < (0.5 * max_freq))

                        low_energy = np.sum(fft_mag[low_mask]**2) if low_mask.any() else 0.0
                        mid_energy = np.sum(fft_mag[mid_mask]**2) if mid_mask.any() else 0.0

                        seg_feats.extend([dom_freq_val, spectral_centroid, low_energy, mid_energy])

                    sample_feats.extend(seg_feats)
            all_features.append(sample_feats)
        return np.array(all_features)
    

############################################
# ScalerTransformer
############################################

class ScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='standard'):
        self.method = method

    def fit(self, X, y=None):
        if self.method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        return self.scaler.transform(X)