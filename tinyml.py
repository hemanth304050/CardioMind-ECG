import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Dict, Any, Tuple
from model import CardioMindECG

class TinyMLProfiler:
    """
    Simulates and profiles model compression (pruning, quantization) and 
    benchmarks deployment performance on edge microcontrollers and CPUs.
    """
    def __init__(self):
        # Baseline hardware specs mapping
        # Keys: hardware name, Values: (latency_multiplier, power_mw, base_latency_ms)
        self.hardware_profiles = {
            "NVIDIA A100 GPU (Cloud)": {
                "base_latency_ms": 3.4,
                "power_mw": 300000.0,
                "latency_scaling": 1.0,
                "quant_scaling": 0.8,
                "cortex_overhead": 1.0
            },
            "Raspberry Pi 4 (Edge CPU)": {
                "base_latency_ms": 35.0,
                "power_mw": 3200.0,
                "latency_scaling": 1.2,
                "quant_scaling": 0.35, # quantized runs much faster
                "cortex_overhead": 1.0
            },
            "ARM Cortex-M7 (STM32H7)": {
                "base_latency_ms": 120.0,
                "power_mw": 42.0,
                "latency_scaling": 1.5,
                "quant_scaling": 0.20,
                "cortex_overhead": 1.4 # no floating point optimization overhead
            },
            "ARM Cortex-M4 (STM32F4)": {
                "base_latency_ms": 280.0,
                "power_mw": 18.0,
                "latency_scaling": 1.8,
                "quant_scaling": 0.15,
                "cortex_overhead": 1.8
            }
        }

    def count_parameters(self, model: nn.Module) -> int:
        """
        Calculates the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def simulate_pruning(self, model: CardioMindECG, prune_ratio: float = 0.5) -> CardioMindECG:
        """
        Simulates magnitude-based weight pruning.
        Sets weights below the threshold percentile to 0.
        """
        pruned_model = copy.deepcopy(model)
        pruned_model.eval()
        
        with torch.no_grad():
            for name, param in pruned_model.named_parameters():
                # Only prune weights of Conv1d and Linear layers
                if "weight" in name and ("conv" in name or "linear" in name or "layer" in name or "proj" in name or "head" in name):
                    flat_param = param.cpu().numpy().flatten()
                    threshold = np.percentile(np.abs(flat_param), prune_ratio * 100)
                    
                    # Apply mask
                    mask = torch.abs(param) > threshold
                    param.data.mul_(mask.float())
                    
        return pruned_model

    def simulate_int8_quantization(self, model: CardioMindECG) -> Tuple[CardioMindECG, float]:
        """
        Simulates INT8 weight quantization.
        Quantizes weights from Float32 to INT8, measures clipping noise, and 
        returns the quantized model along with accuracy retention simulation loss.
        """
        quant_model = copy.deepcopy(model)
        quant_model.eval()
        
        noise_sum = 0.0
        weight_count = 0
        
        with torch.no_grad():
            for name, param in quant_model.named_parameters():
                if "weight" in name and ("conv" in name or "linear" in name or "layer" in name or "proj" in name or "head" in name):
                    w = param.cpu().numpy()
                    
                    # Scale and zero-point estimation (symmetric quantization)
                    max_val = np.max(np.abs(w))
                    if max_val < 1e-7:
                        continue
                        
                    scale = max_val / 127.0
                    
                    # Quantize to INT8 range [-128, 127]
                    w_quant = np.round(w / scale)
                    w_quant = np.clip(w_quant, -128, 127)
                    
                    # Dequantize back to Float32 to simulate loss/noise
                    w_dequant = w_quant * scale
                    
                    # Record quantization noise (mean squared error)
                    noise_sum += np.sum((w - w_dequant) ** 2)
                    weight_count += w.size
                    
                    # Assign dequantized weights back to parameters (simulating accuracy effect)
                    param.copy_(torch.FloatTensor(w_dequant))
                    
        quant_noise = noise_sum / max(1, weight_count)
        return quant_model, float(quant_noise)

    def benchmark_deployment(self, 
                             model: CardioMindECG, 
                             pruning_ratio: float = 0.0, 
                             is_quantized: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Benchmarks size, latency, power, and simulated accuracy retention 
        for the four hardware targets under the given compression configuration.
        """
        # Base stats of model
        base_param_count = self.count_parameters(model)
        float32_size_mb = base_param_count * 4 / (1024 * 1024)
        
        # Calculate compressed size
        # Pruning reduces active parameters (sparse representation) but might not reduce size unless using sparse formats.
        # We assume structured/zipped index compression.
        size_multiplier = (1.0 - 0.7 * pruning_ratio)  # 70% effective storage saving from pruning
        if is_quantized:
            size_multiplier *= 0.25  # 4x reduction for INT8
            
        model_size_mb = float32_size_mb * size_multiplier
        
        # Estimate accuracy retention
        # Pruning and quantization cause minor drops in performance
        acc_retention = 100.0
        
        # Pruning penalty
        if pruning_ratio > 0:
            if pruning_ratio <= 0.3:
                acc_retention -= pruning_ratio * 1.5
            elif pruning_ratio <= 0.6:
                acc_retention -= 1.0 + (pruning_ratio - 0.3) * 4.0
            else:
                acc_retention -= 3.0 + (pruning_ratio - 0.6) * 15.0
                
        # Quantization penalty
        if is_quantized:
            acc_retention -= 0.6
            
        acc_retention = max(20.0, acc_retention)
        
        results = {}
        for hw_name, profile in self.hardware_profiles.items():
            # Estimate Latency (ms)
            # Compression speeds up execution. Quantization accelerates MCU operations via SIMD/CMSIS-NN.
            # Pruning reduces FLOPS if supported by kernel, we assume 30% latency benefit from 60% pruning.
            prune_speedup = 1.0 - (0.4 * pruning_ratio)
            
            if is_quantized:
                speed_factor = profile["quant_scaling"] * prune_speedup
            else:
                speed_factor = profile["latency_scaling"] * prune_speedup * profile["cortex_overhead"]
                
            latency_ms = profile["base_latency_ms"] * speed_factor
            
            # Estimate Power (mW)
            # Quantized memory access takes less power
            power_factor = 0.7 if is_quantized else 1.0
            power_mw = profile["power_mw"] * power_factor
            
            # RAM/Flash footprint
            ram_footprint_kb = model_size_mb * 1024.0
            
            results[hw_name] = {
                "size_mb": float(model_size_mb),
                "ram_kb": float(ram_footprint_kb),
                "latency_ms": float(latency_ms),
                "power_mw": float(power_mw),
                "acc_retention_pct": float(acc_retention),
                "energy_per_inf_uj": float(latency_ms * power_mw / 1000.0) # Energy in microJoules
            }
            
        return results

# Quick test
if __name__ == "__main__":
    model = CardioMindECG()
    profiler = TinyMLProfiler()
    
    params = profiler.count_parameters(model)
    print(f"Total model parameters: {params:,}")
    print(f"Base Float32 model size: {params*4/(1024*1024):.2f} MB")
    
    # Run simulated pruning (50%)
    pruned = profiler.simulate_pruning(model, 0.5)
    
    # Run simulated quantization
    quant, noise = profiler.simulate_int8_quantization(model)
    print(f"Quantization noise (MSE): {noise:.6e}")
    
    # Benchmark deployment
    bench = profiler.benchmark_deployment(model, pruning_ratio=0.50, is_quantized=True)
    for hw, stats in bench.items():
        print(f"Hardware: {hw}")
        print(f"  Size: {stats['size_mb']:.2f} MB | Latency: {stats['latency_ms']:.2f} ms | Power: {stats['power_mw']:.1f} mW | Acc: {stats['acc_retention_pct']:.1f}%")
