import sys
import os
import shutil
import zipfile
import torch
import ultralytics.nn.modules.block
from ultralytics.nn.modules.block import AAttn
from ultralytics import YOLO

# Monkey patch AAttn to handle old model structure (missing qkv, has qk and v)
original_forward = AAttn.forward

def patched_forward(self, x):
    if hasattr(self, 'qkv'):
        return original_forward(self, x)
    else:
        # Fallback for old model structure
        B, C, H, W = x.shape
        N = H * W
        
        # Compute qk and v
        # Assuming self.qk and self.v exist based on inspection
        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x).flatten(2).transpose(1, 2)
        
        if self.area > 1:
            # qk has 2 * C channels (query + key)
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
            
        q, k = (
            qk.view(B, N, self.num_heads, self.head_dim * 2)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim], dim=2)
        )
        v = (
            v.view(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 3, 1)
        )
        
        # The rest is identical to original forward
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)

# Apply patch
AAttn.forward = patched_forward

def export_model():
    model_path = '/Users/husainraja/Downloads/best/cattle_model.pt'
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please run detect_cow.py first to create it.")
        return

    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
        
        print("Starting export to TFLite...")
        # Export the model
        # int8 quantization is often good for mobile, but let's start with standard float32 or float16
        # format='tflite'
        model.export(format='tflite', int8=True)
        
        print("Export complete.")
        
    except Exception as e:
        print(f"Export failed: {e}")

if __name__ == "__main__":
    export_model()
