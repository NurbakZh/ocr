import os
from ultralytics import YOLO

def main():
    try:
        # Get absolute path to data.yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_yaml_path = os.path.join(current_dir, "data.yaml")

        # Load a YOLOv8 model
        model = YOLO("yolov8s.pt")

        # Train the model with absolute path
        model.train(data=data_yaml_path, epochs=100, imgsz=640)

        # Validate the model
        model.val()

        # Save the model
        model.save("best_model.pt")

        # Try to export the model, but don't fail if it errors
        try:
            model.export(format="onnx")
        except Exception as e:
            print(f"ONNX export failed (this is not critical): {str(e)}")
            print("The model is still trained and saved successfully.")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
