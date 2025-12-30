import sys
import os
from ultralytics import YOLO

def run_tflite(image_path):
    # Path to the TFLite model generated earlier
    model_path = 'best_saved_model/best_float32.tflite'
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please run export_tflite.py first.")
        return

    print(f"Loading TFLite model from {model_path}...")
    try:
        # Ultralytics supports loading .tflite models directly
        model = YOLO(model_path, task='detect')
        
        print(f"Running inference on {image_path}...")
        # Run inference
        results = model(image_path)
        
        for result in results:
            if len(result.boxes) == 0:
                print("No detections.")
                continue
                
            # Process detections
            for box in result.boxes:
                class_id = int(box.cls)
                # Note: TFLite models might lose class names metadata depending on export.
                # If names are missing, we might need to map them manually if we know them.
                # But Ultralytics usually handles metadata.
                if model.names:
                    class_name = model.names[class_id]
                else:
                    class_name = f"Class {class_id}"
                    
                confidence = float(box.conf)
                print(f"Detected: {class_name} (Confidence: {confidence:.2f})")

    except Exception as e:
        print(f"Error during TFLite inference: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_tflite.py <image_path>")
    else:
        run_tflite(sys.argv[1])
