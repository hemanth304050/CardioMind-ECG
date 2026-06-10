import argparse
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Any, Tuple, List

# Custom imports
from simulator import ECGSimulator
from preprocessing import ECGPreprocessor
from model import CardioMindECG, MultiTaskLossWrapper

# Map scenarios to target labels
SCENARIO_TARGETS = {
    "normal":             {"arr": 0, "stress": 0, "val": 0, "ar": 1},  # NSR, Low, Positive, Low
    "arrhythmia_af":      {"arr": 1, "stress": 1, "val": 1, "ar": 0},  # AFib, Med, Negative, High
    "arrhythmia_pvc":     {"arr": 2, "stress": 1, "val": 1, "ar": 0},  # PVC, Med, Negative, High
    "arrhythmia_vt":      {"arr": 3, "stress": 2, "val": 1, "ar": 0},  # VT, High, Negative, High
    "stress_high":        {"arr": 0, "stress": 2, "val": 1, "ar": 0},  # NSR, High, Negative, High
    "stress_low":         {"arr": 0, "stress": 0, "val": 0, "ar": 1},  # NSR, Low, Positive, Low
    "emotion_happy":      {"arr": 0, "stress": 0, "val": 0, "ar": 0},  # NSR, Low, Positive, High
    "emotion_sad":        {"arr": 0, "stress": 1, "val": 1, "ar": 1}   # NSR, Med, Negative, Low
}

def generate_dataset(num_sequences: int = 40) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Generates a physiological synthetic dataset of segmented beat sequences and HRV feature vectors.
    """
    fs = 256
    seq_len = 60
    beat_len = 150
    
    sim = ECGSimulator(fs=fs)
    pre = ECGPreprocessor(fs=fs)
    
    X_beats = []
    X_hrv = []
    y_arr = []
    y_stress = []
    y_val = []
    y_ar = []
    
    scenarios = list(SCENARIO_TARGETS.keys())
    
    print(f"Generating {num_sequences} synthetic ECG sequence blocks...")
    for idx in range(num_sequences):
        # Sample scenarios in a balanced loop
        scen = scenarios[idx % len(scenarios)]
        targets = SCENARIO_TARGETS[scen]
        
        # High noise on some samples to force model robustness
        noise = 0.04 + 0.08 * (idx % 3)
        
        # Generate enough duration to guarantee ~70 beats
        duration = 75.0 if scen == "stress_low" else 55.0
        data = sim.generate_scenario(scen, duration_seconds=duration, noise_level=noise)
        
        # Preprocess
        filtered = pre.denoise(data["raw"])
        peaks = pre.detect_r_peaks(filtered)
        beats, valid_peaks = pre.segment_beats(filtered, peaks, beat_len=beat_len)
        hrv = pre.extract_hrv_features(peaks)
        
        # Ensure exact sequence length 60
        if len(beats) < seq_len:
            if len(beats) == 0:
                # Fallback if no beats detected
                beats = np.zeros((seq_len, beat_len))
            else:
                beats = np.pad(beats, ((0, seq_len - len(beats)), (0, 0)), 'edge')
        else:
            beats = beats[:seq_len]
            
        hrv_list = [
            hrv["mean_hr"], hrv["sdnn"], hrv["rmssd"], hrv["pnn50"],
            hrv["lf"], hrv["hf"], hrv["lf_hf_ratio"],
            hrv["sd1"], hrv["sd2"], hrv["sd1_sd2_ratio"], hrv["sample_entropy"]
        ]
        
        X_beats.append(beats)
        X_hrv.append(hrv_list)
        y_arr.append(targets["arr"])
        y_stress.append(targets["stress"])
        y_val.append(targets["val"])
        y_ar.append(targets["ar"])
        
    X_beats_tensor = torch.FloatTensor(np.array(X_beats))  # [N, 60, 150]
    X_hrv_tensor = torch.FloatTensor(np.array(X_hrv))      # [N, 11]
    
    # Scale HRV features (Z-score normalize for stability)
    hrv_mean = X_hrv_tensor.mean(dim=0, keepdim=True)
    hrv_std = X_hrv_tensor.std(dim=0, keepdim=True) + 1e-6
    X_hrv_tensor = (X_hrv_tensor - hrv_mean) / hrv_std
    
    labels = {
        "arrhythmia": torch.LongTensor(y_arr),
        "stress": torch.LongTensor(y_stress),
        "valence": torch.LongTensor(y_val),
        "arousal": torch.LongTensor(y_ar)
    }
    
    return X_beats_tensor, X_hrv_tensor, labels

def train_model_pipeline(epochs: int = 4, num_sequences: int = 40, device: str = "cpu"):
    """
    Trains the CardioMind-ECG model using joint training and dynamic uncertainty loss weighting.
    """
    print(f"Training started on device: {device}")
    
    # 1. Generate Dataset
    X_beats, X_hrv, labels = generate_dataset(num_sequences)
    
    dataset = TensorDataset(
        X_beats, X_hrv, 
        labels["arrhythmia"], labels["stress"], 
        labels["valence"], labels["arousal"]
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 2. Build Model & Loss Wrapper
    model = CardioMindECG()
    model.to(device)
    loss_wrapper = MultiTaskLossWrapper(num_tasks=4)
    loss_wrapper.to(device)
    
    # Joint optimizer for model params + uncertainty log variances
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.001},
        {'params': loss_wrapper.parameters(), 'lr': 0.005}
    ], weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        task_losses_accum = {"arrhythmia": 0.0, "stress": 0.0, "valence": 0.0, "arousal": 0.0}
        
        for batch in dataloader:
            b_beats, b_hrv, b_arr, b_stress, b_val, b_ar = [t.to(device) for t in batch]
            
            optimizer.zero_grad()
            
            # Forward pass
            preds, _ = model(b_beats, b_hrv)
            
            # Compute independent losses
            l_arr = criterion(preds["arrhythmia"], b_arr)
            l_stress = criterion(preds["stress"], b_stress)
            l_val = criterion(preds["valence"], b_val)
            l_ar = criterion(preds["arousal"], b_ar)
            
            losses = {
                "arrhythmia": l_arr,
                "stress": l_stress,
                "valence": l_val,
                "arousal": l_ar
            }
            
            # Combine losses dynamically
            total_loss, scaled_losses = loss_wrapper(losses)
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            for k in task_losses_accum:
                task_losses_accum[k] += losses[k].item()
                
        # Print metrics
        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {epoch_loss/len(dataloader):.4f} | "
              f"Arr: {task_losses_accum['arrhythmia']/len(dataloader):.3f} | "
              f"Str: {task_losses_accum['stress']/len(dataloader):.3f} | "
              f"Val: {task_losses_accum['valence']/len(dataloader):.3f} | "
              f"Ar: {task_losses_accum['arousal']/len(dataloader):.3f}")
        
    # Save Model Weights
    torch.save(model.state_dict(), "cardio_mind_model.pth")
    print("Model training complete. Weights saved to 'cardio_mind_model.pth'")

def evaluate_model_pipeline(device: str = "cpu"):
    """
    Evaluates a saved CardioMind-ECG model on a validation dataset.
    """
    model_path = "cardio_mind_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Train the model first.")
        sys.exit(1)
        
    model = CardioMindECG()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Generate test dataset
    X_beats, X_hrv, labels = generate_dataset(num_sequences=16)
    
    with torch.no_grad():
        X_beats, X_hrv = X_beats.to(device), X_hrv.to(device)
        preds, _ = model(X_beats, X_hrv)
        
        # Calculate Accuracies
        for task, out in preds.items():
            true_labels = labels[task].numpy()
            pred_labels = torch.argmax(out, dim=1).cpu().numpy()
            acc = np.mean(true_labels == pred_labels) * 100
            print(f"Task '{task.capitalize()}' Evaluation Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CardioMind-ECG Command Line Coordinator")
    parser.add_argument("--train", action="store_true", help="Train the model on synthetic physiological data")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model on a test set")
    parser.add_argument("--run-app", action="store_true", help="Run the Streamlit interactive dashboard (default)")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Default to running the application if no arguments are provided
    if not (args.train or args.evaluate or args.run_app):
        args.run_app = True
        
    if args.train:
        train_model_pipeline(epochs=5, num_sequences=40, device=device)
    elif args.evaluate:
        evaluate_model_pipeline(device=device)
    elif args.run_app:
        print("Launching Streamlit dashboard app.py...")
        # Execute Streamlit run app.py
        os.system("streamlit run app.py")
