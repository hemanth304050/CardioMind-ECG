import numpy as np
import scipy.signal as signal
from typing import Tuple, List, Dict, Any

class ECGPreprocessor:
    """
    Denoising, R-peak detection, beat segmentation, and HRV feature extraction.
    """
    def __init__(self, fs: int = 256):
        self.fs = fs
        
    def denoise(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Butterworth bandpass filter (0.5 - 45 Hz) to remove baseline wander and high-frequency noise.
        """
        nyq = self.fs / 2
        low = 0.5 / nyq
        high = 45.0 / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, raw_signal)
        return filtered_signal

    def detect_r_peaks(self, clean_signal: np.ndarray) -> np.ndarray:
        """
        Finds R-peaks using a modified Pan-Tompkins algorithm structure.
        """
        # Step 1: Bandpass filter (5-15 Hz) to isolate QRS energy
        nyq = self.fs / 2
        low = 5.0 / nyq
        high = 15.0 / nyq
        b_bp, a_bp = signal.butter(3, [low, high], btype='band')
        qrs_filtered = signal.filtfilt(b_bp, a_bp, clean_signal)
        
        # Step 2: Derivative filter
        # Highlight slopes of QRS: y[n] = 1/8 (2x[n] + x[n-1] - x[n-3] - 2x[n-4])
        # A simple derivative diff is also effective:
        deriv = np.diff(qrs_filtered)
        deriv = np.append(deriv, 0)  # Maintain shape
        
        # Step 3: Squaring function
        squared = deriv ** 2
        
        # Step 4: Moving window integration (150ms window)
        win_len = int(0.150 * self.fs)
        if win_len % 2 == 0:
            win_len += 1
        integrated = np.convolve(squared, np.ones(win_len) / win_len, mode='same')
        
        # Step 5: Adaptive threshold peak detection
        # We can use scipy's find_peaks on the integrated signal
        # Min peak distance is 350ms (physiological limit for 170 BPM)
        min_dist = int(0.350 * self.fs)
        
        # Adaptive height threshold: 1.5 * average of integrated signal, or 15% of max
        height_thresh = max(0.01, 0.15 * np.max(integrated))
        
        peaks, _ = signal.find_peaks(integrated, distance=min_dist, height=height_thresh)
        
        # Refine peak locations: search local maxima in raw/clean signal around integrated peak
        refined_peaks = []
        search_win = int(0.100 * self.fs)  # 100ms search window around peak
        
        for peak in peaks:
            start = max(0, peak - search_win)
            end = min(len(clean_signal), peak + search_win)
            if start >= end:
                continue
            # R-peak is the local maximum (or minimum if inverted, but we assume positive peaks)
            local_max_idx = start + np.argmax(clean_signal[start:end])
            refined_peaks.append(local_max_idx)
            
        return np.unique(refined_peaks)

    def segment_beats(self, ecg_signal: np.ndarray, r_peaks: np.ndarray, 
                      beat_len: int = 150) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segments ECG beats centered around R-peaks.
        Returns a numpy array of shape [num_beats, beat_len] and active peak indices.
        Window is approx. -200ms (50 samples at 256Hz) to +400ms (100 samples at 256Hz).
        """
        half_len = beat_len // 2
        # Use asymmetric window: 1/3 before R-peak, 2/3 after R-peak
        pre_samples = int(0.200 * self.fs)  # 200 ms before R-peak
        post_samples = beat_len - pre_samples
        
        beats = []
        valid_peaks = []
        
        for peak in r_peaks:
            start = peak - pre_samples
            end = peak + post_samples
            
            # Ensure boundaries are within signal limits
            if start >= 0 and end <= len(ecg_signal):
                beat = ecg_signal[start:end]
                # Normalize amplitude of each beat (Z-score normalize)
                std_val = np.std(beat)
                if std_val > 1e-4:
                    beat = (beat - np.mean(beat)) / std_val
                else:
                    beat = beat - np.mean(beat)
                
                beats.append(beat)
                valid_peaks.append(peak)
                
        return np.array(beats), np.array(valid_peaks)

    def extract_hrv_features(self, r_peaks: np.ndarray) -> Dict[str, float]:
        """
        Computes HRV features from R-peaks.
        Handles edge cases (too few peaks) gracefully by returning baseline values.
        """
        # Calculate RR intervals in milliseconds
        rr_intervals = np.diff(r_peaks) / self.fs * 1000.0
        
        # Baseline dictionary
        hrv = {
            "mean_hr": 70.0,
            "sdnn": 50.0,
            "rmssd": 35.0,
            "pnn50": 10.0,
            "lf": 500.0,
            "hf": 400.0,
            "lf_hf_ratio": 1.25,
            "sd1": 25.0,
            "sd2": 70.0,
            "sd1_sd2_ratio": 0.35,
            "sample_entropy": 1.2
        }
        
        if len(rr_intervals) < 3:
            return hrv
            
        # 1. Time Domain Features
        hrv["mean_hr"] = 60000.0 / np.mean(rr_intervals)
        hrv["sdnn"] = float(np.std(rr_intervals))
        
        diff_rr = np.diff(rr_intervals)
        hrv["rmssd"] = float(np.sqrt(np.mean(diff_rr ** 2)))
        
        nn50 = np.sum(np.abs(diff_rr) > 50.0)
        hrv["pnn50"] = float(nn50 / len(diff_rr) * 100.0)
        
        # 2. Nonlinear Features (Poincaré Plot)
        # SD1 corresponds to short term variability (RMSSD direction)
        # SD2 corresponds to long term variability
        # x = rr[:-1], y = rr[1:]
        # sd1 = std((x - y) / sqrt(2)), sd2 = std((x + y) / sqrt(2))
        x = rr_intervals[:-1]
        y = rr_intervals[1:]
        diff_xy = (x - y) / np.sqrt(2)
        sum_xy = (x + y) / np.sqrt(2)
        hrv["sd1"] = float(np.std(diff_xy))
        hrv["sd2"] = float(np.std(sum_xy))
        hrv["sd1_sd2_ratio"] = hrv["sd1"] / hrv["sd2"] if hrv["sd2"] > 0 else 0.0
        
        # 3. Frequency Domain Features (Lomb-Scargle Periodogram)
        # Excellent for unevenly spaced RR intervals
        try:
            # Time of R-peaks in seconds relative to first peak
            t_peaks = (r_peaks[1:] - r_peaks[0]) / self.fs
            
            # Target frequencies (Hz)
            # LF: 0.04 to 0.15 Hz
            # HF: 0.15 to 0.4 Hz
            freqs_lf = np.linspace(0.04, 0.15, 50)
            freqs_hf = np.linspace(0.15, 0.40, 50)
            
            # Center RR intervals
            rr_detrend = rr_intervals - np.mean(rr_intervals)
            
            # Compute Lomb-Scargle powers
            p_lf = signal.lombscargle(t_peaks, rr_detrend, freqs_lf * 2 * np.pi)
            p_hf = signal.lombscargle(t_peaks, rr_detrend, freqs_hf * 2 * np.pi)
            
            # Integrate powers (areas under curve)
            lf_power = float(np.trapz(p_lf, freqs_lf))
            hf_power = float(np.trapz(p_hf, freqs_hf))
            
            # Keep positive values
            hrv["lf"] = max(1.0, lf_power)
            hrv["hf"] = max(1.0, hf_power)
            hrv["lf_hf_ratio"] = hrv["lf"] / hrv["hf"]
        except Exception:
            # Fallbacks if Lomb-Scargle fails
            pass
            
        # 4. Sample Entropy (Optimized implementation)
        hrv["sample_entropy"] = float(self._sample_entropy(rr_intervals, m=2, r=0.2 * np.std(rr_intervals)))
        
        return hrv

    def _sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 15.0) -> float:
        """
        Calculates Sample Entropy of a 1D sequence.
        Handles zero-variance and boundary conditions.
        """
        N = len(data)
        if N < m + 1 or r <= 0:
            return 1.2
            
        def _count(m_val):
            # Form templates of length m_val
            templates = np.array([data[i : i + m_val] for i in range(N - m_val + 1)])
            # Count how many pairs are closer than r in Chebyshev distance
            count = 0
            for i in range(len(templates)):
                diffs = np.max(np.abs(templates - templates[i]), axis=1)
                count += np.sum(diffs < r) - 1  # Subtract self-comparison
            return count
            
        num_a = _count(m + 1)
        num_b = _count(m)
        
        if num_a == 0 or num_b == 0:
            return 2.5  # Return high entropy if no templates match (highly irregular)
            
        return -np.log(num_a / num_b)

# Fast test to ensure it works
if __name__ == "__main__":
    from simulator import ECGSimulator
    sim = ECGSimulator(fs=256)
    data = sim.generate_scenario("normal", duration_seconds=15.0, noise_level=0.08)
    
    pre = ECGPreprocessor(fs=256)
    clean = pre.denoise(data["raw"])
    peaks = pre.detect_r_peaks(clean)
    beats, valid_peaks = pre.segment_beats(clean, peaks)
    hrv = pre.extract_hrv_features(peaks)
    
    print("Preprocessing test completed successfully!")
    print(f"Detected R-peaks: {len(peaks)}")
    print(f"Segmented beats shape: {beats.shape}")
    print(f"HRV RMSSD: {hrv['rmssd']:.2f} ms, LF/HF: {hrv['lf_hf_ratio']:.2f}")
