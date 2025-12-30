import os
import sys
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

def repair_model(model_path='best.pt', source_dir='.'):
    """
    Repairs the model by zipping the unzipped files into a valid PyTorch archive.
    PyTorch saves models as a zip file where the contents are usually in a subdirectory (often 'archive').
    """
    print(f"Checking model at {model_path}...")
    
    # Check if model exists and is a valid file
    if os.path.exists(model_path) and os.path.isfile(model_path):
        try:
            # Try loading to see if it's valid
            YOLO(model_path)
            print("Model is valid.")
            return True
        except Exception as e:
            print(f"Model exists but failed to load: {e}. Attempting repair...")
    
    print("Creating valid model archive...")
    
    # Files expected in the unzipped directory
    required_files = ['data.pkl', 'version', 'byteorder', '.format_version', '.storage_alignment']
    required_dirs = ['data', '.data']
    
    # Verify source files exist
    for f in required_files:
        if not os.path.exists(os.path.join(source_dir, f)):
            print(f"Error: Missing required file '{f}' in {source_dir}")
            return False
            
    try:
        with zipfile.ZipFile(model_path, 'w', zipfile.ZIP_STORED) as zipf:
            # The root folder inside the zip
            archive_root = 'archive'
            
            # Add files
            for f in required_files:
                src = os.path.join(source_dir, f)
                dst = os.path.join(archive_root, f)
                # Create ZipInfo to handle timestamps
                zinfo = zipfile.ZipInfo(dst)
                zinfo.date_time = (2025, 1, 1, 0, 0, 0)
                zinfo.compress_type = zipfile.ZIP_STORED
                with open(src, 'rb') as file_data:
                    zipf.writestr(zinfo, file_data.read())
                
            # Add directories
            for d in required_dirs:
                src_dir = os.path.join(source_dir, d)
                if os.path.exists(src_dir):
                    for root, _, files in os.walk(src_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Calculate relative path from source_dir
                            rel_path = os.path.relpath(file_path, source_dir)
                            # Destination path inside zip
                            dst_path = os.path.join(archive_root, rel_path)
                            
                            # Create ZipInfo
                            zinfo = zipfile.ZipInfo(dst_path)
                            zinfo.date_time = (2025, 1, 1, 0, 0, 0)
                            zinfo.compress_type = zipfile.ZIP_STORED
                            
                            with open(file_path, 'rb') as file_data:
                                zipf.writestr(zinfo, file_data.read())
                            
        print(f"Successfully created {model_path}")
        return True
        
    except Exception as e:
        print(f"Failed to create model archive: {e}")
        return False

def detect_cow(image_path):
    model_path = 'best.pt'
    
    # Ensure model is ready
    if not repair_model(model_path):
        print("Could not load or create a valid model file.")
        return

    try:
        # Load the model
        model = YOLO(model_path)
        
        # Run inference
        results = model(image_path)
        
        for result in results:
            # Check if any detections
            if len(result.boxes) == 0:
                print("No cow detected.")
                continue
                
            # Get the box with highest confidence
            best_box = max(result.boxes, key=lambda x: x.conf)
            class_id = int(best_box.cls)
            class_name = model.names[class_id]
            confidence = float(best_box.conf)
            
            print(f"Detected: {class_name} (Confidence: {confidence:.2f})")
            
            # Specific logic for user request
            if "Gir" in class_name and "Sahiwal" in class_name:
                 print("Result: Gir X Sahiwal")
            elif "Gir" in class_name:
                 print("Result: Gir")
            else:
                 print(f"Result: {class_name}")

    except Exception as e:
        print(f"Error during detection: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_cow.py <image_path>")
    else:
        detect_cow(sys.argv[1])
