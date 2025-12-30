import sys
import json
import os
from ultralytics import YOLO

def predict(image_path):
    try:
        # Load the model
        # Assuming the model files are in ../models/yolo_model
        # If 'best' was a directory containing weights, we might need to point to specific weights
        # But if it's a standard YOLO export, pointing to the dir might work or we look for .pt
        
        model_dir = os.path.join(os.path.dirname(__file__), '../')
        
        # Check if there's a .pt file in the directory, otherwise try loading the directory
        # Based on previous analysis, it seemed to be an unzipped structure. 
        # Ultralytics usually expects a .pt file. 
        # If the user provided a directory that WAS a .pt file (unzipped), we might have issues.
        # However, the user said "we have a model analyzed that model fast" and it had data.pkl.
        # This structure (data.pkl, data/) is typical of a PyTorch save.
        # Let's try loading it as a path.
        
        model = YOLO(model_dir)

        # Run inference
        results = model(image_path)

        # Process results
        output = []
        for result in results:
            for box in result.boxes:
                output.append({
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy.tolist()[0]
                })

        # Return JSON
        print(json.dumps({"success": True, "predictions": output}))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No image path provided"}))
    else:
        predict(sys.argv[1])
