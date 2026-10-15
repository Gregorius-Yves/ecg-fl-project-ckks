"""
Pipeline Lengkap Pembersihan Sinyal EKG (12-Lead Multi-Label FL)
Urutan pra-pemrosesan yang ideal untuk menyiapkan tensor training:
1. Filtering: Jalankan bandpass filter 0.5 - 45 Hz pada sampling rate asli (misal 500 Hz).
2. Resampling: Downsample ke 100 Hz (scipy.signal.resample) agar komputasi ringan (10s = 1000 sampel).
3. Normalisasi: Terapkan Z-score normalization per lead ((x - mu) / (sigma + eps)) agar gradien konvergen stabil di PyTorch.
"""

import numpy as np
from scipy import signal


class ECGPreprocessor:
    def __init__(
        self,
        orig_rate: int = 500,
        target_rate: int = 100,
        target_duration: int = 10,
        lowcut: float = 0.5,
        highcut: float = 45.0,
        filter_order: int = 3,
        norm_method: str = "zscore"
    ):
        self.orig_rate = orig_rate
        self.target_rate = target_rate
        self.target_duration = target_duration
        self.target_samples = target_rate * target_duration  # 10s @ 100 Hz = 1000 samples
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.norm_method = norm_method.lower()

        # Desain koefisien Butterworth Bandpass 0.5 - 45 Hz pada sampling rate asli (500 Hz)
        nyq = 0.5 * self.orig_rate
        low = self.lowcut / nyq
        high = self.highcut / nyq
        self.b_band, self.a_band = signal.butter(self.filter_order, [low, high], btype='bandpass')

    def apply_filtering(self, ecg_data: np.ndarray) -> np.ndarray:
        """
        Langkah 1: Bandpass filter 0.5 - 45 Hz pada sampling rate asli.
        Menggunakan vectorization axis=-1 untuk efisiensi tinggi pada seluruh 12 lead sekaligus.
        """
        ecg_data = np.nan_to_num(ecg_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            return signal.filtfilt(self.b_band, self.a_band, ecg_data, axis=-1).astype(np.float32)
        except Exception:
            return ecg_data.astype(np.float32)

    def apply_resampling(self, ecg_data: np.ndarray, fs: int = None) -> np.ndarray:
        """
        Langkah 2: Downsample ke 100 Hz (target_samples = 1000 sampel untuk 10s).
        Jika durasi sinyal mentah bukan 10 detik, sinyal di-crop atau zero-padding ke tepat target_samples.
        """
        fs = fs or self.orig_rate
        num_leads, current_samples = ecg_data.shape
        expected_resampled_len = int(round(current_samples * (self.target_rate / fs)))

        if expected_resampled_len != current_samples and expected_resampled_len > 0:
            try:
                resampled = signal.resample(ecg_data, expected_resampled_len, axis=-1).astype(np.float32)
            except Exception:
                resampled = ecg_data
        else:
            resampled = ecg_data

        # Standardisasi panjang ke tepat target_samples (1000 sampel = 10s @ 100 Hz)
        cur_len = resampled.shape[-1]
        if cur_len == self.target_samples:
            return resampled

        output = np.zeros((num_leads, self.target_samples), dtype=np.float32)
        if cur_len < self.target_samples:
            output[:, :cur_len] = resampled
        else:
            start = (cur_len - self.target_samples) // 2
            output = resampled[:, start:start + self.target_samples]
        return output

    def apply_normalization(self, ecg_data: np.ndarray) -> np.ndarray:
        """
        Langkah 3: Z-score normalization per lead ((x - mean) / (std + eps)) atau Min-Max.
        """
        if self.norm_method == "minmax":
            min_val = np.min(ecg_data, axis=-1, keepdims=True)
            max_val = np.max(ecg_data, axis=-1, keepdims=True)
            denom = (max_val - min_val) + 1e-8
            return (ecg_data - min_val) / denom
        else:
            mean = np.mean(ecg_data, axis=-1, keepdims=True)
            std = np.std(ecg_data, axis=-1, keepdims=True) + 1e-8
            return (ecg_data - mean) / std

    def process_record(self, raw_ecg: np.ndarray, fs: int = None) -> np.ndarray:
        """
        Pipeline Lengkap:
        Raw ECG (12, N) -> [1. Filtering 0.5-45 Hz] -> [2. Resampling ke 100 Hz (1000)] -> [3. Normalisasi Z-score]
        Output dimension: (12, 1000)
        """
        filtered = self.apply_filtering(raw_ecg)
        resampled = self.apply_resampling(filtered, fs=fs)
        normalized = self.apply_normalization(resampled)
        return normalized
