"""
ECG Signal Preprocessing & Normalization Module
Standardizes 12-lead raw ECG recordings to exact dimensions: (12, 5000) (10s @ 500 Hz).
Applies Butterworth Bandpass, Notch Filter (50/60Hz), and Length Standardization.
"""

import numpy as np
from scipy import signal


class ECGPreprocessor:
    def __init__(self, target_rate=500, target_duration=10, lowcut=0.05, highcut=150.0, notch_freq=50.0):
        self.target_rate = target_rate
        self.target_duration = target_duration
        self.target_samples = target_rate * target_duration  # 5000 samples
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq

    def butter_bandpass(self, order=3):
        nyq = 0.5 * self.target_rate
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = signal.butter(order, [low, high], btype='bandpass')
        return b, a

    def butter_notch(self, quality_factor=30.0):
        b, a = signal.iirnotch(self.notch_freq, quality_factor, self.target_rate)
        return b, a

    def apply_filters(self, ecg_data: np.ndarray) -> np.ndarray:
        """
        ecg_data shape: (12, N)
        """
        b_band, a_band = self.butter_bandpass()
        b_notch, a_notch = self.butter_notch()

        filtered_ecg = np.zeros_like(ecg_data, dtype=np.float32)
        for lead_idx in range(ecg_data.shape[0]):
            lead_signal = ecg_data[lead_idx]
            # Bandpass filter
            lead_signal = signal.filtfilt(b_band, a_band, lead_signal)
            # Notch filter
            lead_signal = signal.filtfilt(b_notch, a_notch, lead_signal)
            filtered_ecg[lead_idx] = lead_signal

        return filtered_ecg

    def standardize_length(self, ecg_data: np.ndarray) -> np.ndarray:
        """
        Zero-padding if signal < 5000 samples, center cropping if signal > 5000 samples.
        ecg_data: (12, N) -> returns: (12, 5000)
        """
        num_leads, current_samples = ecg_data.shape
        if current_samples == self.target_samples:
            return ecg_data

        standardized = np.zeros((num_leads, self.target_samples), dtype=np.float32)

        if current_samples < self.target_samples:
            # Zero-padding di akhir
            standardized[:, :current_samples] = ecg_data
        else:
            # Center cropping
            start = (current_samples - self.target_samples) // 2
            standardized = ecg_data[:, start:start + self.target_samples]

        return standardized

    def normalize_zscore(self, ecg_data: np.ndarray) -> np.ndarray:
        """Normalisasi per-lead z-score: (x - mean) / (std + eps)"""
        mean = np.mean(ecg_data, axis=-1, keepdims=True)
        std = np.std(ecg_data, axis=-1, keepdims=True) + 1e-8
        return (ecg_data - mean) / std

    def process_record(self, raw_ecg: np.ndarray) -> np.ndarray:
        """
        Pipeline lengkap: Filtering -> Length Standardization -> Z-score Normalization.
        Output shape: (12, 5000)
        """
        filtered = self.apply_filters(raw_ecg)
        length_adjusted = self.standardize_length(filtered)
        normalized = self.normalize_zscore(length_adjusted)
        return normalized
