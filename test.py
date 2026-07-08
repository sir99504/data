#!/usr/bin/env python3
# check_gradients.py
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_model_gradients():
    print("=== CHECKING MODEL GRADIENTS ===")
    
    
    cfg = {
        'seq_len': 7,
        'spatial_offset': 3,
        'input_size': 10,
        'hidden_size': 128,
        'kernel_size': 3,
        'dropout_rate': 0.15,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }
    
    device = torch.device(cfg['device'])
    
    
    from model import GraphConvLSTMModel
    
    
    model = GraphConvLSTMModel(cfg).to(device)
    
    
    batch_size = 2
    seq_len = cfg['seq_len']
    in_channels = cfg['input_size']
    spatial_size = 2 * cfg['spatial_offset'] + 1
    
  
    test_input = torch.rand(batch_size, seq_len, in_channels, spatial_size, spatial_size).to(device) * 2 - 1  
    
    
    target = torch.rand(batch_size, 1, spatial_size, spatial_size).to(device)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test input range: [{test_input.min().item():.4f}, {test_input.max().item():.4f}]")
    print(f"Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")
    
    
    model.train()
    output = model(test_input)
    print(f"\nModel output shape: {output.shape}")
    print(f"Model output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
   
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(output, target)
    print(f"Loss: {loss.item():.6f}")
    
   
    model.zero_grad()
    loss.backward()
    
    
    print("\n=== GRADIENT ANALYSIS ===")
    
    total_params = 0
    zero_gradients = 0
    small_gradients = 0
    large_gradients = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            grad_std = param.grad.std().item()
            
            if grad_mean == 0:
                zero_gradients += 1
            elif grad_mean < 1e-8:
                small_gradients += 1
            elif grad_mean > 1e2:
                large_gradients += 1
            
            if "senet" in name or "attn" in name or "conv" in name:
                print(f"{name:40} grad mean: {grad_mean:.2e}, std: {grad_std:.2e}")
        
        total_params += 1
    
    print(f"\nTotal parameters: {total_params}")
    print(f"Zero gradients: {zero_gradients}")
    print(f"Very small gradients (<1e-8): {small_gradients}")
    print(f"Very large gradients (>1e2): {large_gradients}")
    
    
    if zero_gradients > total_params * 0.5:
        print("? WARNING: More than 50% parameters have zero gradients!")
    if large_gradients > 0:
        print("? WARNING: Some gradients are very large (potential gradient explosion)")
    
    return model, output, loss

if __name__ == "__main__":
    check_model_gradients()