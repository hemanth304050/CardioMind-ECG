import numpy as np
import scipy.signal as signal
from typing import Tuple, List, Dict, Any

class ECGSimulator:
    """
    Physiology-based synthetic ECG signal generator simulating wearable data.
    Can inject various arrhythmias, stress levels, emotional states, and noise.
    """
    def __init__(self, fs: int = 256):
        self.fs = fs  # Sampling rate in Hz
        
    def _generate_beat_template(self, beat_type: str, state: Dict[str, Any], t_rr: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generates a single ECG beat template centered around the R-peak (t=0).
        Returns a time series of length fs * t_rr and an annotation dictionary.
        """
        # Duration of the beat in samples
        n_samples = int(t_rr * self.fs)
        t = np.linspace(-t_rr * 0.35, t_rr * 0.65, n_samples)  # R-peak at index ~35%
        
        # Base wave parameters (amplitudes, offsets in seconds, widths in seconds)
        # P, Q, R, S, T waves
        p_amp, p_off, p_w = 0.15, -0.18, 0.025
        q_amp, q_off, q_w = -0.12, -0.02, 0.005
        r_amp, r_off, r_w = 1.20, 0.0, 0.010
        s_amp, s_off, s_w = -0.25, 0.025, 0.008
        t_amp, t_off, t_w = 0.35, 0.22, 0.040
        st_level = 0.0  # ST segment level
        
        # Apply Stress effects
        stress_level = state.get("stress_level", "low")
        if stress_level == "high":
            # Sympathetic activation: slight ST-segment depression/elevation, taller/smaller T-waves, slightly faster QRS
            st_level = -0.05
            t_amp = 0.25
            r_amp = 1.30
        elif stress_level == "medium":
            st_level = -0.02
            t_amp = 0.30
            
        # Apply Emotion effects
        arousal = state.get("arousal", "low")
        valence = state.get("valence", "positive")
        if arousal == "high":
            t_amp += 0.05
            r_amp += 0.05
        if valence == "negative":
            t_amp -= 0.03
            
        # Override parameters based on arrhythmia type
        if beat_type == "AF":
            # Atrial Fibrillation: No P-wave, replace with f-waves (small fast oscillations)
            p_amp = 0.0
            # Generate f-wave baseline oscillation
            f_freq = np.random.uniform(6.0, 9.0)
            f_amp = np.random.uniform(0.03, 0.06)
            f_wave = f_amp * np.sin(2 * np.pi * f_freq * (t + 0.2))
        else:
            f_wave = np.zeros_like(t)
            
        if beat_type == "PVC":
            # Premature Ventricular Contraction: Wide QRS, giant R, inverted T, no P-wave
            p_amp = 0.0
            q_amp, q_off, q_w = -0.05, -0.04, 0.015
            r_amp, r_off, r_w = 1.80, 0.0, 0.035
            s_amp, s_off, s_w = -0.60, 0.05, 0.025
            t_amp, t_off, t_w = -0.55, 0.26, 0.070
            st_level = -0.15
            
        if beat_type == "VT":
            # Ventricular Tachycardia: Rapid, wide, smooth QRS-T complexes. No P wave.
            p_amp = 0.0
            q_amp = 0.0
            r_amp, r_off, r_w = 1.60, 0.0, 0.045
            s_amp, s_off, s_w = -0.80, 0.06, 0.030
            t_amp, t_off, t_w = -0.40, 0.22, 0.060
            st_level = -0.10

        # Construct individual wave components using Gaussian equations
        def gaussian(x, amp, off, width):
            return amp * np.exp(-((x - off) ** 2) / (2 * (width ** 2)))
        
        p_wave = gaussian(t, p_amp, p_off, p_w) if p_amp != 0 else np.zeros_like(t)
        q_wave = gaussian(t, q_amp, q_off, q_w)
        r_wave = gaussian(t, r_amp, r_off, r_w)
        s_wave = gaussian(t, s_off, s_off, s_w)  # wait, gaussian(t, s_amp, s_off, s_w)
        s_wave = gaussian(t, s_amp, s_off, s_w)
        
        # For T-wave, add ST segment deviation
        # ST segment transition from S to T wave
        st_transition = np.zeros_like(t)
        st_idx = (t > s_off) & (t < t_off - 0.05)
        st_transition[st_idx] = st_level
        
        t_wave = gaussian(t, t_amp, t_off, t_w) + st_level * gaussian(t, 1.0, (s_off + t_off)/2.0, 0.08)
        
        # Combine components
        ecg_beat = p_wave + q_wave + r_wave + s_wave + t_wave + f_wave
        
        # Find local R-peak index (it should be close to t = 0)
        r_idx = np.argmin(np.abs(t))
        
        annotations = {
            "r_peak_idx": r_idx,
            "beat_type": beat_type,
            "p_wave_range": (np.argmin(np.abs(t - (p_off - 2*p_w))), np.argmin(np.abs(t - (p_off + 2*p_w)))),
            "qrs_range": (np.argmin(np.abs(t - (q_off - 2*q_w))), np.argmin(np.abs(t - (s_off + 2*s_w)))),
            "t_wave_range": (np.argmin(np.abs(t - (t_off - 2*t_w))), np.argmin(np.abs(t - (t_off + 2*t_w)))),
        }
        
        return ecg_beat, annotations

    def generate_scenario(self, 
                          scenario_name: str = "normal", 
                          duration_seconds: float = 60.0,
                          noise_level: float = 0.05) -> Dict[str, Any]:
        """
        Generates a continuous ECG signal for a specific physiological scenario.
        Scenarios: 'normal', 'arrhythmia_af', 'arrhythmia_pvc', 'arrhythmia_vt', 'stress_high', 'stress_low', 'emotion_happy', 'emotion_sad'
        """
        # Determine average heart rate and HRV parameters
        state = {"stress_level": "low", "arousal": "low", "valence": "positive"}
        base_bpm = 70.0
        hrv_std = 0.050  # 50 ms SDNN equivalent
        rsa_amp = 0.040  # RSA amplitude in seconds
        
        if scenario_name == "normal":
            pass
        elif scenario_name == "arrhythmia_af":
            base_bpm = 95.0
            hrv_std = 0.150  # Highly irregular RR
            rsa_amp = 0.0
        elif scenario_name == "arrhythmia_pvc":
            base_bpm = 72.0
            hrv_std = 0.040
        elif scenario_name == "arrhythmia_vt":
            base_bpm = 160.0  # Tachycardia
            hrv_std = 0.005  # Highly regular rapid beats
            rsa_amp = 0.0
        elif scenario_name == "stress_high":
            state["stress_level"] = "high"
            state["arousal"] = "high"
            state["valence"] = "negative"
            base_bpm = 105.0
            hrv_std = 0.015  # Depressed SDNN
            rsa_amp = 0.005  # Depressed RSA
        elif scenario_name == "stress_low":
            state["stress_level"] = "low"
            state["arousal"] = "low"
            state["valence"] = "positive"
            base_bpm = 62.0
            hrv_std = 0.065  # Rich SDNN
            rsa_amp = 0.060  # Rich RSA
        elif scenario_name == "emotion_happy":
            state["stress_level"] = "low"
            state["arousal"] = "high"
            state["valence"] = "positive"
            base_bpm = 80.0
            hrv_std = 0.060
            rsa_amp = 0.055
        elif scenario_name == "emotion_sad":
            state["stress_level"] = "medium"
            state["arousal"] = "low"
            state["valence"] = "negative"
            base_bpm = 68.0
            hrv_std = 0.035
            rsa_amp = 0.020
            
        # 1. Generate sequence of R-R intervals
        rr_intervals = []
        total_time = 0.0
        
        # Respiration wave (for RSA simulation)
        resp_freq = 0.25  # 15 breaths per minute
        
        while total_time < duration_seconds + 5.0:
            # RSA component
            rsa = rsa_amp * np.sin(2 * np.pi * resp_freq * total_time)
            
            # Autonomic fluctuations (LF power simulation via a 0.1Hz slow sine)
            lf_fluc = 0.015 * np.sin(2 * np.pi * 0.1 * total_time)
            if state["stress_level"] == "high":
                lf_fluc *= 1.8  # Elevated LF
                
            # Random HRV variation
            rand_var = np.random.normal(0, hrv_std)
            
            # Calculate next RR interval
            rr = (60.0 / base_bpm) + rsa + lf_fluc + rand_var
            
            # Ensure physiological limits
            rr = np.clip(rr, 0.3, 2.0)
            rr_intervals.append(rr)
            total_time += rr
            
        # 2. Assemble ECG waveform beat by beat
        ecg_signal_clean = []
        r_peaks = []
        beat_types = []
        current_sample_idx = 0
        
        for i, rr in enumerate(rr_intervals):
            # Decide beat type
            b_type = "N"  # Normal
            if scenario_name == "arrhythmia_af":
                b_type = "AF"
            elif scenario_name == "arrhythmia_vt":
                b_type = "VT"
            elif scenario_name == "arrhythmia_pvc":
                # Inject a PVC every 8 to 12 beats, but not the very first/last beats
                if i > 2 and i < len(rr_intervals) - 3 and i % 9 == 0:
                    b_type = "PVC"
                    
            # Generate the beat waveform
            # For PVC, we shorten the preceding RR interval and lengthen the succeeding one (compensatory pause)
            if b_type == "PVC":
                rr_actual = rr * 0.70  # Premature
            elif i > 0 and beat_types[-1] == "PVC":
                rr_actual = rr * 1.30  # Compensatory pause
            else:
                rr_actual = rr
                
            beat_wave, beat_annot = self._generate_beat_template(b_type, state, rr_actual)
            
            # Record R-peak index relative to start of continuous signal
            r_peak_loc = current_sample_idx + beat_annot["r_peak_idx"]
            r_peaks.append(r_peak_loc)
            beat_types.append(b_type)
            
            ecg_signal_clean.extend(beat_wave)
            current_sample_idx += len(beat_wave)
            
        ecg_signal_clean = np.array(ecg_signal_clean)
        
        # Truncate to exact duration requested
        max_samples = int(duration_seconds * self.fs)
        ecg_signal_clean = ecg_signal_clean[:max_samples]
        r_peaks = [p for p in r_peaks if p < max_samples]
        beat_types = beat_types[:len(r_peaks)]
        
        # 3. Add Motion Artifacts and Noise
        t_full = np.arange(len(ecg_signal_clean)) / self.fs
        
        # Baseline wander (low frequency drift, e.g., 0.15 Hz respiration + 0.05 Hz body motion)
        baseline_wander = 0.25 * np.sin(2 * np.pi * 0.15 * t_full) + 0.15 * np.cos(2 * np.pi * 0.04 * t_full)
        # Under high noise, baseline wander is larger
        if noise_level > 0.1:
            baseline_wander *= 2.5
            
        # High-frequency muscle noise (EMG) - band-limited noise
        nyq = self.fs / 2
        # High pass filter white noise at 35Hz to simulate muscle noise
        b_emg, a_emg = signal.butter(4, 30.0 / nyq, btype='high')
        raw_noise = np.random.normal(0, 1.0, len(ecg_signal_clean))
        emg_noise = signal.filtfilt(b_emg, a_emg, raw_noise)
        emg_noise = (emg_noise / np.std(emg_noise)) * noise_level
        
        # Combine
        ecg_signal_raw = ecg_signal_clean + baseline_wander + emg_noise
        
        return {
            "time": t_full,
            "raw": ecg_signal_raw,
            "clean_ground_truth": ecg_signal_clean,
            "baseline_wander": baseline_wander,
            "noise": emg_noise,
            "r_peaks": np.array(r_peaks),
            "beat_types": beat_types,
            "stress_level": state["stress_level"],
            "arousal": state["arousal"],
            "valence": state["valence"],
            "fs": self.fs
        }

# Fast test to ensure it works
if __name__ == "__main__":
    sim = ECGSimulator(fs=256)
    data = sim.generate_scenario("arrhythmia_pvc", duration_seconds=10.0, noise_level=0.05)
    print("Simulator test completed successfully!")
    print(f"Signal length: {len(data['raw'])} samples, R-peaks: {len(data['r_peaks'])}")
    print(f"Beat types: {data['beat_types']}")
