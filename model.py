import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

class ResNet1DBlock(nn.Module):
    """
    1D Residual Block for morphological features.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 3):
        super(ResNet1DBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class CNNMorphologicalEncoder(nn.Module):
    """
    CNN Morphological Encoder based on 1D ResNet-18.
    Processes single ECG beats (shape: [batch_size, 1, beat_len]) to 256-dim embeddings.
    """
    def __init__(self, input_len: int = 150, embedding_dim: int = 256):
        super(CNNMorphologicalEncoder, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet Residual Blocks with increasing receptive fields
        self.layer1 = ResNet1DBlock(64, 64, stride=1, kernel_size=3)
        self.layer2 = ResNet1DBlock(64, 128, stride=2, kernel_size=5)  # Output len: beat_len / 2
        self.layer3 = ResNet1DBlock(128, 256, stride=2, kernel_size=7) # Output len: beat_len / 4
        self.layer4 = ResNet1DBlock(256, embedding_dim, stride=2, kernel_size=9) # Output len: beat_len / 8
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save intermediate feature maps for Grad-CAM
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # [batch_size, embedding_dim]
        return x

class TemporalAttentionModule(nn.Module):
    """
    Computes attention weights over the sequence of beat representations.
    Inputs: [batch_size, seq_len, lstm_hidden * 2] (Bi-LSTM outputs)
    Outputs: context vector [batch_size, lstm_hidden * 2] and weights [batch_size, seq_len]
    """
    def __init__(self, hidden_dim: int):
        super(TemporalAttentionModule, self).__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch, seq_len, hidden_dim]
        scores = self.attn_net(x)  # [batch, seq_len, 1]
        attn_weights = F.softmax(scores, dim=1)  # [batch, seq_len, 1]
        
        # Compute context vector (weighted sum)
        context = torch.sum(attn_weights * x, dim=1)  # [batch, hidden_dim]
        
        return context, attn_weights.squeeze(-1)

class CardioMindECG(nn.Module):
    """
    Unified CardioMind-ECG network for Arrhythmia, Stress, and Emotion recognition.
    """
    def __init__(self, beat_len: int = 150, hrv_dim: int = 11):
        super(CardioMindECG, self).__init__()
        
        # 1. Morphological CNN Encoder (processes single beats)
        self.morph_encoder = CNNMorphologicalEncoder(input_len=beat_len, embedding_dim=256)
        
        # 2. Temporal Bi-LSTM Encoder (processes sequence of beats)
        # Input size: 256 (morphology embedding size)
        # Hidden size: 256, Bidirectional: output size = 512
        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0.3
        )
        
        # 3. Attention Module
        self.attention = TemporalAttentionModule(hidden_dim=512)
        
        # 4. HRV Projection
        self.hrv_proj = nn.Sequential(
            nn.Linear(hrv_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        # Fusion dimensionality: 512 (LSTM attention) + 64 (HRV projected) = 576
        fusion_dim = 512 + 64
        
        # 5. Multi-Task Heads
        # A: Arrhythmia detection (5 classes: Normal, AFib, PVC, VT, Bradycardia)
        self.arrhythmia_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )
        
        # B: Stress prediction (3 classes: Low, Medium, High)
        self.stress_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )
        
        # C: Emotion heads (Binary Valence & Arousal represented as 2 classes each)
        self.valence_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, beat_seq: torch.Tensor, hrv_feats: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        beat_seq: Tensor of shape [batch_size, seq_len, beat_len]
        hrv_feats: Tensor of shape [batch_size, hrv_dim]
        """
        batch_size, seq_len, beat_len = beat_seq.size()
        
        # Flatten sequence to pass through morphological CNN
        # shape: [batch_size * seq_len, 1, beat_len]
        flat_beats = beat_seq.view(batch_size * seq_len, 1, beat_len)
        flat_embeddings = self.morph_encoder(flat_beats)
        
        # Reshape back to sequence
        # shape: [batch_size, seq_len, 256]
        seq_embeddings = flat_embeddings.view(batch_size, seq_len, 256)
        
        # Pass through Bi-LSTM
        lstm_out, _ = self.lstm(seq_embeddings)  # [batch_size, seq_len, 512]
        
        # Temporal attention aggregation
        context_vector, attn_weights = self.attention(lstm_out)  # context: [batch_size, 512]
        
        # Project HRV features
        hrv_projected = self.hrv_proj(hrv_feats)  # [batch_size, 64]
        
        # Concatenate features
        fused_features = torch.cat([context_vector, hrv_projected], dim=1)  # [batch_size, 576]
        
        # Multi-task predictions
        predictions = {
            "arrhythmia": self.arrhythmia_head(fused_features),
            "stress": self.stress_head(fused_features),
            "valence": self.valence_head(fused_features),
            "arousal": self.arousal_head(fused_features)
        }
        
        return predictions, attn_weights

class MultiTaskLossWrapper(nn.Module):
    """
    Combines task losses using dynamic uncertainty weighting (Kendall et al., 2018).
    """
    def __init__(self, num_tasks: int = 4):
        super(MultiTaskLossWrapper, self).__init__()
        # Learnable log variances for dynamic loss scaling
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Order of tasks: arrhythmia, stress, valence, arousal
        task_names = ["arrhythmia", "stress", "valence", "arousal"]
        
        total_loss = 0.0
        scaled_losses = {}
        
        for idx, task in enumerate(task_names):
            loss = losses[task]
            log_var = self.log_vars[idx]
            precision = torch.exp(-log_var)
            
            # Weighted loss equation
            task_loss = 0.5 * precision * loss + 0.5 * log_var
            total_loss += task_loss
            scaled_losses[task] = float(task_loss.item())
            
        return total_loss, scaled_losses

# Fast test to ensure architecture shapes are correct
if __name__ == "__main__":
    model = CardioMindECG()
    # Batch size 4, sequence of 60 beats, each beat has 150 samples
    dummy_seq = torch.randn(4, 60, 150)
    # 11 HRV features
    dummy_hrv = torch.randn(4, 11)
    
    preds, weights = model(dummy_seq, dummy_hrv)
    print("Model forward pass verified!")
    for task, out in preds.items():
        print(f"  Task '{task}' output shape: {out.shape}")
    print(f"  Attention weights shape: {weights.shape}")
