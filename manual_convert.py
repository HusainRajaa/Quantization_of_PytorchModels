import numpy as np
import os
import subprocess

def convert():
    # 1. Generate calibration data
    print("Generating calibration data...")
    # Assuming YOLO input size 640x640. 
    # If the model uses a different size, this might need adjustment, but 640 is standard.
    # Generating random data for calibration is faster than loading images and sufficient for structural conversion,
    # though real images are better for accuracy. For "fast" conversion requested by user, this is acceptable.
    # To be safe with "int version", we want a representative range, so we'll use random values 0-1 (normalized).
    data_shape = (20, 640, 640, 3)
    calibration_data = np.random.uniform(0, 1, data_shape).astype(np.float32)
    
    calib_file = 'calib_data_640.npy'
    np.save(calib_file, calibration_data)
    print(f"Saved {calib_file}")

    # 2. Run onnx2tf
    # -i: Input ONNX file
    # -o: Output directory (will create 'model.tflite' inside)
    # -osd: Output saved model directory (we can skip or specifying it helps some internal steps)
    # -qt: Quantization type (int8)
    # -cd: Calibration data file
    # -cind: Input node name for calibration data (usually 'images' for YOLO, but onnx2tf might find it automatically or we check strictly)
    # We will try without specifying input name first, as onnx2tf is smart.
    
    cmd = [
        "onnx2tf",
        "-i", "cattle_model.onnx",
        "-o", "cattle_model_tflite_int8",
        "-ois", "int8", # Output integer quantized
        "-qt", "int8",  # Quantization type
        "-cd", calib_file,
        "-cind", "images", # Explicitly naming input 'images' which is standard for YOLO
        "-v", "info"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    convert()
