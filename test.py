from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

def expand_coordinates(x1, y1, x2, y2, padding=4, max_h=None, max_w=None):
    """Expand coordinates by padding pixels while respecting image boundaries"""
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    
    if max_w is not None:
        x2 = min(max_w, x2 + padding)
    else:
        x2 = x2 + padding
        
    if max_h is not None:
        y2 = min(max_h, y2 + padding)
    else:
        y2 = y2 + padding
        
    return x1, y1, x2, y2

# Load the trained model
model = YOLO("best_model.pt")

# Function to perform inference on an image
def detect_objects(image_path):
    # Create output directory if it doesn't exist
    output_dir = "detected_parts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]  # Get image dimensions
    
    # Perform inference
    results = model(image)[0]  # Get the first result
    
    # Create a copy of the image for drawing
    annotated_image = image.copy()
    
    # Dictionary to store highest confidence detections for each class
    best_detections = {}
    
    # Process all detections
    for box in results.boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        class_name = model.names[class_id]
        
        # Update best detection for this class if confidence is higher
        if class_name not in best_detections or confidence > best_detections[class_name]['conf']:
            best_detections[class_name] = {
                'box': box.xyxy[0].cpu().numpy(),  # Get box coordinates
                'conf': confidence
            }
    
    # Draw the best detection for each class and save cropped images
    for class_name, detection in best_detections.items():
        x1, y1, x2, y2 = map(int, detection['box'])
        
        # Expand coordinates by 4 pixels while respecting image boundaries
        x1, y1, x2, y2 = expand_coordinates(x1, y1, x2, y2, padding=4, max_h=height, max_w=width)
        
        conf = detection['conf']
        
        # Draw rectangle
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f'{class_name} {conf:.2f}'
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated_image, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Crop the detected part with expanded coordinates
        cropped_image = image[y1:y2, x1:x2]
        
        # Save the cropped image
        output_path = os.path.join(output_dir, f"{class_name}_{conf:.2f}.jpg")
        cv2.imwrite(output_path, cropped_image)
        
        # Print detection
        print(f"Detected {class_name} with confidence: {conf:.2f}")
        print(f"Saved cropped image to: {output_path}")

    # Save the annotated full image
    annotated_output_path = os.path.join(output_dir, "annotated_full.jpg")
    cv2.imwrite(annotated_output_path, annotated_image)
    print(f"Saved annotated full image to: {annotated_output_path}")

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Path to the image you want to perform inference on
image_path = 'lol.jpg'

try:
    # Detect objects in the image
    detect_objects(image_path)
except Exception as e:
    print(f"Error occurred: {str(e)}")
