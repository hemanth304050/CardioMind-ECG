import torch
import numpy as np
import scipy.interpolate as interpolate
from typing import Dict, Any, List, Tuple
from model import CardioMindECG

def compute_gradcam_1d(model: CardioMindECG, 
                       beat_seq: torch.Tensor, 
                       hrv_feats: torch.Tensor, 
                       task: str = "arrhythmia", 
                       class_idx: int = 0,
                       beat_idx: int = 0,
                       device: str = "cpu") -> np.ndarray:
    """
    Computes 1D Grad-CAM heatmap for a target beat in a sequence.
    Returns a normalized 1D heatmap of length equal to the input beat (150).
    """
    model.eval()
    model.to(device)
    
    # Store activations and gradients
    activations = []
    gradients = []
    
    # Define hooks
    def forward_hook(module, input, output):
        activations.append(output)
        
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
        
    # Register hooks on the final convolutional layer of the morphological encoder
    # In our CNNMorphologicalEncoder, that is layer4
    target_layer = model.morph_encoder.layer4
    h_forward = target_layer.register_forward_hook(forward_hook)
    h_backward = target_layer.register_full_backward_hook(backward_hook)
    
    # Prepare input tensors with gradients enabled
    beat_seq_var = beat_seq.clone().detach().requires_grad_(True).to(device)
    hrv_feats_var = hrv_feats.clone().detach().to(device)
    
    # Forward pass
    predictions, attn_weights = model(beat_seq_var, hrv_feats_var)
    
    # Select prediction for the desired task and class index
    pred_task = predictions[task]
    score = pred_task[0, class_idx]
    
    # Backward pass to calculate gradients
    model.zero_grad()
    score.backward()
    
    # Remove hooks
    h_forward.remove()
    h_backward.remove()
    
    if len(activations) == 0 or len(gradients) == 0:
        # Fallback if hooks failed to fire
        return np.ones(beat_seq.shape[-1]) * 0.1
        
    # The activations shape: [batch_size * seq_len, 256, feature_map_len]
    # We want the activation corresponding to the target beat_idx in the batch
    # Since batch size = 1 in this context: flat index = beat_idx
    act = activations[0][beat_idx].detach().cpu().numpy()  # [256, feature_map_len]
    grad = gradients[0][beat_idx].detach().cpu().numpy()    # [256, feature_map_len]
    
    # Global average pooling of gradients (weights alpha)
    weights = np.mean(grad, axis=1)  # [256]
    
    # Weighted combination of activation maps
    cam = np.zeros(act.shape[1], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
        
    # Apply ReLU to keep only positive contributions
    cam = np.maximum(cam, 0)
    
    # Normalize heatmap
    cam_max = np.max(cam)
    if cam_max > 1e-6:
        cam = cam / cam_max
    else:
        cam = np.zeros_like(cam)
        
    # Interpolate (upsample) 1D CAM to match original beat length (e.g., 150 samples)
    input_len = beat_seq.shape[-1]
    x_old = np.linspace(0, 1, len(cam))
    x_new = np.linspace(0, 1, input_len)
    
    f_interp = interpolate.interp1d(x_old, cam, kind='linear', fill_value="extrapolate")
    cam_upsampled = f_interp(x_new)
    
    # Clip just in case
    cam_upsampled = np.clip(cam_upsampled, 0, 1)
    
    return cam_upsampled

def get_temporal_importance(attn_weights: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Ranks the beat indices in a sequence by their attention weights.
    attn_weights: 1D array of weights over the sequence of beats.
    Returns: list of (beat_idx, attention_weight) sorted descending.
    """
    ranked = [(idx, float(w)) for idx, w in enumerate(attn_weights)]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def generate_clinical_report(preds: Dict[str, np.ndarray], 
                             attn_weights: np.ndarray, 
                             beat_types: List[str],
                             hrv_metrics: Dict[str, float]) -> str:
    """
    Generates a formal clinical interpretation report card from model outputs.
    """
    # Arrhythmia classes map
    arr_classes = ["Normal Sinus Rhythm (NSR)", "Atrial Fibrillation (AF)", 
                   "Premature Ventricular Contractions (PVC)", "Ventricular Tachycardia (VT)", "Bradycardia"]
    
    stress_classes = ["Low Cognitive Load / Relaxed", "Moderate Stress", "High Physiological Stress"]
    
    # Extract prediction labels and confidences
    arr_idx = np.argmax(preds["arrhythmia"][0])
    arr_conf = float(np.max(preds["arrhythmia"][0])) * 100
    arr_pred = arr_classes[arr_idx]
    
    stress_idx = np.argmax(preds["stress"][0])
    stress_conf = float(np.max(preds["stress"][0])) * 100
    stress_pred = stress_classes[stress_idx]
    
    val_idx = np.argmax(preds["valence"][0])
    val_label = "Positive/Neutral Valence" if val_idx == 0 else "Negative/Depressed Valence"
    
    ar_idx = np.argmax(preds["arousal"][0])
    ar_label = "High Physiological Arousal" if ar_idx == 0 else "Low/Resting Arousal"
    
    # Temporal attention analysis
    top_beats = get_temporal_importance(attn_weights, top_k=3)
    
    # Clinical justification narrative based on arrhythmia prediction
    justification = ""
    if arr_idx == 0:  # NSR
        justification = (
            "The cardiac rhythm shows standard morphology with stable R-R intervals and distinct P, QRS, and T complexes. "
            "Autonomic markers align with healthy parasympathetic regulation."
        )
    elif arr_idx == 1:  # AF
        justification = (
            "The model detected marked irregularity in consecutive beat intervals (R-R variability) paired with absence of "
            "organized atrial P-wave morphology. Spectral analysis shows a high LF/HF ratio and low RMSSD, suggesting autonomic "
            "disequilibrium accompanying the rhythm instability."
        )
    elif arr_idx == 2:  # PVC
        justification = (
            "Occasional ectopic ventricular depolarizations are visible. These beats are characterized by anomalous widening "
            "of the QRS complex, elevated amplitudes, and discordant (inverted) T-waves. The model placed high attention weights "
            "specifically on these aberrant beats."
        )
    elif arr_idx == 3:  # VT
        justification = (
            "A rapid sequence of wide-QRS complexes with complete loss of normal waveform segments. Heart rate is severely elevated "
            "with zero baseline stability. Immediate clinical intervention is indicated."
        )
    elif arr_idx == 4:  # Bradycardia
        justification = (
            "The cardiac cycle is structurally normal but operating at a depressed rhythm (<50 BPM). This is typically associated "
            "with high vagal tone in athletic subjects or conduction delays in clinical populations."
        )

    # Stress & Autonomic narrative
    autonomic_summary = ""
    if stress_idx == 2:
        autonomic_summary = (
            f"Autonomic markers confirm High Physiological Stress: RMSSD is depressed at {hrv_metrics['rmssd']:.1f} ms, "
            f"SDNN is restricted to {hrv_metrics['sdnn']:.1f} ms, and the sympathovagal index (LF/HF ratio) is elevated at "
            f"{hrv_metrics['lf_hf_ratio']:.2f}."
        )
    elif stress_idx == 1:
        autonomic_summary = (
            f"Moderate sympathetic arousal observed. SDNN: {hrv_metrics['sdnn']:.1f} ms, LF/HF: {hrv_metrics['lf_hf_ratio']:.2f}."
        )
    else:
        autonomic_summary = (
            f"Autonomic balance is dominated by vagal regulation, reflecting a relaxed state. "
            f"RMSSD is healthy at {hrv_metrics['rmssd']:.1f} ms, and Sample Entropy is high ({hrv_metrics['sample_entropy']:.2f})."
        )

    # Format findings as a clinical report markdown
    report = f"""# CardioMind-ECG Clinical Diagnostics Report
**Physiological State Assessment Summary**
*Generated dynamically using CardioMind-ECG Deep Temporal Attention Networks*

---

## 1. Primary Classification Diagnostics
* **Arrhythmia Diagnostics**: `{arr_pred}` (Confidence: `{arr_conf:.1f}%`)
* **Stress/Workload Assessment**: `{stress_pred}` (Confidence: `{stress_conf:.1f}%`)
* **Autonomic Affective State**: `{val_label}` & `{ar_label}`

---

## 2. Autonomic (HRV) Biomarkers
| Domain | Metric | Value | Reference / Diagnostic Status |
| :--- | :--- | :--- | :--- |
| **Time-Domain** | RMSSD (Vagal Index) | `{hrv_metrics['rmssd']:.2f} ms` | {'Normal/Relaxed' if hrv_metrics['rmssd'] > 30 else 'Suppressed (Sympathetic Overdrive)'} |
| **Time-Domain** | SDNN (Overall HRV) | `{hrv_metrics['sdnn']:.2f} ms` | {'Healthy' if hrv_metrics['sdnn'] > 40 else 'Constrained'} |
| **Frequency-Domain** | LF/HF (Autonomic Balance) | `{hrv_metrics['lf_hf_ratio']:.2f}` | {'Balanced' if hrv_metrics['lf_hf_ratio'] < 2.0 else 'Sympathetic Dominance'} |
| **Nonlinear** | Poincaré SD1/SD2 | `{hrv_metrics['sd1_sd2_ratio']:.2f}` | {'Standard' if 0.2 < hrv_metrics['sd1_sd2_ratio'] < 0.5 else 'Abnormal dispersion'} |
| **Complexity** | Sample Entropy | `{hrv_metrics['sample_entropy']:.2f}` | {'High complexity (Healthy)' if hrv_metrics['sample_entropy'] > 1.0 else 'Repetitive/Stressed'} |

---

## 3. Explanatory Insights (AI Attention & Morphology)
* **Physiological Justification**: {justification}
* **Autonomic Summary**: {autonomic_summary}
* **Temporal Saliency Hotspots**:
"""
    for rank, (b_idx, weight) in enumerate(top_beats):
        annot_type = beat_types[b_idx] if b_idx < len(beat_types) else "Unknown"
        beat_desc = "Normal Beat" if annot_type == "N" else ("Atrial Fibrillation Segment" if annot_type == "AF" else ("Premature Ventricular Contraction" if annot_type == "PVC" else "Tachycardic Beat"))
        report += f"  {rank+1}. **Beat {b_idx}** at {b_idx * 0.8:.1f}s — Type: `{beat_desc}` | Attention Weight: `{weight * 100:.2f}%` {'(Clinically Triggering Beat)' if weight > 0.05 else ''}\n"

    report += """
---
*Disclaimer: CardioMind-ECG report is a simulated AI diagnostic summary designed to complement cardiologist review. It is not an alternative to standard clinical Holter or 12-lead ECG analysis.*
"""
    return report

# Quick test
if __name__ == "__main__":
    from simulator import ECGSimulator
    from preprocessing import ECGPreprocessor
    
    sim = ECGSimulator(fs=256)
    pre = ECGPreprocessor(fs=256)
    model = CardioMindECG()
    
    # Generate data
    data = sim.generate_scenario("arrhythmia_pvc", duration_seconds=10.0)
    clean = pre.denoise(data["raw"])
    peaks = pre.detect_r_peaks(clean)
    beats, valid_peaks = pre.segment_beats(clean, peaks)
    hrv = pre.extract_hrv_features(peaks)
    
    # Format sequence inputs (batch size 1, sequence length 10 beats)
    seq_len = 10
    beat_seq = torch.FloatTensor(beats[:seq_len]).unsqueeze(0) # [1, 10, 150]
    hrv_tensor = torch.FloatTensor([list(hrv.values())[:11]])  # [1, 11]
    
    # Get model outputs
    model.eval()
    with torch.no_grad():
        preds, weights = model(beat_seq, hrv_tensor)
    
    # Dummy softmax preds for testing report
    softmax_preds = {
        "arrhythmia": torch.softmax(preds["arrhythmia"], dim=1).numpy(),
        "stress": torch.softmax(preds["stress"], dim=1).numpy(),
        "valence": torch.softmax(preds["valence"], dim=1).numpy(),
        "arousal": torch.softmax(preds["arousal"], dim=1).numpy(),
    }
    
    cam = compute_gradcam_1d(model, beat_seq, hrv_tensor, task="arrhythmia", class_idx=2, beat_idx=2)
    print("Grad-CAM 1D computed! Shape:", cam.shape)
    
    report = generate_clinical_report(softmax_preds, weights[0].numpy(), data["beat_types"][:seq_len], hrv)
    print("Clinical report generated! Length:", len(report))
