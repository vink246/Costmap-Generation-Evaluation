"""
Model Analysis Script for Costmap Generation Models

This script provides comprehensive analysis of model performance including:
- Parameter count
- FLOPs (Floating Point Operations)
- Inference latency
- Memory usage
- Model size

Usage:
    python scripts/analyze_model.py --config configs/train_nyu_unet.yaml
"""

import argparse
import time
import torch
import torch.nn as nn
import importlib
import yaml
import os
import sys
from typing import Dict, Any, Tuple
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def calculate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    Calculate FLOPs for the model.
    Note: This is a simplified calculation and may not be 100% accurate for all operations.
    """
    def flop_count_hook(module, input, output):
        if isinstance(module, nn.Conv2d):
            # Conv2d FLOPs = output_elements * kernel_size * input_channels
            output_elements = output.numel()
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            input_channels = module.in_channels
            module.flops = output_elements * kernel_size * input_channels
            
        elif isinstance(module, nn.ConvTranspose2d):
            # ConvTranspose2d FLOPs = output_elements * kernel_size * input_channels
            output_elements = output.numel()
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            input_channels = module.in_channels
            module.flops = output_elements * kernel_size * input_channels
            
        elif isinstance(module, nn.Linear):
            # Linear FLOPs = input_features * output_features
            input_features = input[0].numel() // input[0].shape[0]  # per sample
            output_features = output.numel() // output.shape[0]  # per sample
            module.flops = input_features * output_features * input[0].shape[0]
            
        else:
            module.flops = 0
    
    # Register hooks
    hooks = []
    for module in model.modules():
        hook = module.register_forward_hook(flop_count_hook)
        hooks.append(hook)
    
    # Forward pass with dummy input
    dummy_input = torch.randn(1, *input_shape).to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Sum up FLOPs
    total_flops = sum(module.flops for module in model.modules())
    return total_flops


def measure_inference_latency(model: nn.Module, input_shape: Tuple[int, ...], 
                            num_warmup: int = 10, num_iterations: int = 100) -> Dict[str, float]:
    """Measure inference latency in milliseconds."""
    device = next(model.parameters()).device
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    # Measure latency
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency_ms = (total_time / num_iterations) * 1000
    
    return {
        'avg_latency_ms': avg_latency_ms,
        'total_time_s': total_time,
        'iterations': num_iterations,
        'fps': 1000 / avg_latency_ms if avg_latency_ms > 0 else 0
    }


def measure_memory_usage(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
    """Measure memory usage in MB."""
    device = next(model.parameters()).device
    
    # Clear cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Measure model memory
    model_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    model_memory_mb = model_memory / (1024 * 1024)
    
    # Measure inference memory
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    
    if device.type == 'cuda':
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.empty_cache()
    else:
        # For CPU, we can't easily measure peak memory, so estimate
        peak_memory_mb = model_memory_mb * 2  # Rough estimate
    
    return {
        'model_memory_mb': model_memory_mb,
        'peak_inference_memory_mb': peak_memory_mb,
        'estimated_total_memory_mb': model_memory_mb + peak_memory_mb
    }


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    return {
        'model_size_mb': total_size_mb,
        'parameter_size_mb': param_size / (1024 * 1024),
        'buffer_size_mb': buffer_size / (1024 * 1024)
    }


def analyze_model(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """Comprehensive model analysis."""
    print("Analyzing model...")
    
    # Parameter count
    print("  Counting parameters...")
    param_info = count_parameters(model)
    
    # FLOPs calculation
    print("  Calculating FLOPs...")
    flops = calculate_flops(model, input_shape)
    
    # Inference latency
    print("  Measuring inference latency...")
    latency_info = measure_inference_latency(model, input_shape)
    
    # Memory usage
    print("  Measuring memory usage...")
    memory_info = measure_memory_usage(model, input_shape)
    
    # Model size
    print("  Calculating model size...")
    size_info = get_model_size(model)
    
    return {
        'parameters': param_info,
        'flops': flops,
        'latency': latency_info,
        'memory': memory_info,
        'size': size_info
    }


def print_analysis_report(analysis: Dict[str, Any], model_name: str = "Model"):
    """Print a formatted analysis report."""
    print(f"\n{'='*60}")
    print(f"{model_name} Analysis Report")
    print(f"{'='*60}")
    
    # Parameters
    print(f"\nParameters:")
    print(f"  Total Parameters:      {analysis['parameters']['total_parameters']:,}")
    print(f"  Trainable Parameters: {analysis['parameters']['trainable_parameters']:,}")
    print(f"  Non-trainable:        {analysis['parameters']['non_trainable_parameters']:,}")
    
    # FLOPs
    print(f"\nComputational Complexity:")
    print(f"  FLOPs:                {analysis['flops']:,}")
    print(f"  FLOPs (M):            {analysis['flops'] / 1e6:.2f}M")
    print(f"  FLOPs (G):            {analysis['flops'] / 1e9:.2f}G")
    
    # Latency
    print(f"\nInference Performance:")
    print(f"  Avg Latency:          {analysis['latency']['avg_latency_ms']:.2f} ms")
    print(f"  FPS:                  {analysis['latency']['fps']:.1f}")
    print(f"  Iterations:           {analysis['latency']['iterations']}")
    
    # Memory
    print(f"\nMemory Usage:")
    print(f"  Model Memory:         {analysis['memory']['model_memory_mb']:.2f} MB")
    print(f"  Peak Inference:      {analysis['memory']['peak_inference_memory_mb']:.2f} MB")
    print(f"  Estimated Total:      {analysis['memory']['estimated_total_memory_mb']:.2f} MB")
    
    # Size
    print(f"\nModel Size:")
    print(f"  Total Size:           {analysis['size']['model_size_mb']:.2f} MB")
    print(f"  Parameter Size:       {analysis['size']['parameter_size_mb']:.2f} MB")
    print(f"  Buffer Size:          {analysis['size']['buffer_size_mb']:.2f} MB")
    
    print(f"\n{'='*60}")


def load_model_from_config(config_path: str):
    """Load model from configuration file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model_module = cfg.get('model_module', 'src.models.unet')
    model_class = cfg.get('model_class', 'UNet')
    model_kwargs = cfg.get('model_kwargs', {})
    
    # Import model class
    ModelClass = getattr(importlib.import_module(model_module), model_class)
    model = ModelClass(**model_kwargs)
    
    return model, cfg


def main():
    parser = argparse.ArgumentParser(description='Analyze model performance and characteristics')
    parser.add_argument('--config', required=True, help='Path to training config YAML file')
    
    args = parser.parse_args()
    
    # Configuration
    config_path = args.config
    input_shape = (4, 256, 256)  # RGB+D input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Input shape: {input_shape}")
    
    # Load model from config
    print(f"Loading model from config: {config_path}")
    model, cfg = load_model_from_config(config_path)
    model_name = cfg.get('model_class', 'Model')
    
    # Move model to device
    model = model.to(device)
    
    # Analyze model
    analysis = analyze_model(model, input_shape)
    
    # Print report
    print_analysis_report(analysis, model_name)

    
if __name__ == '__main__':
    main()
