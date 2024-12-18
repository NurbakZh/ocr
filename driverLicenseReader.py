import cv2
import pytesseract
import json
import os
import numpy as np

def preprocess_categories_image(image):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast
        gray = cv2.equalizeHist(gray)
        
        # Resize image to make it larger
        scale_factor = 2
        enlarged = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(enlarged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        return denoised
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return image

def extract_text_from_image(image_path, filename):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return ""
    
    try:
        if 'Categories' in filename:
            # Special processing for categories
            processed_image = preprocess_categories_image(image)
            # Configure tesseract for license categories
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDE123'
            extracted_text = pytesseract.image_to_string(processed_image, 
                                                       config=custom_config,
                                                       lang='eng+rus+kaz')
            # Clean up the text
            extracted_text = ''.join(c for c in extracted_text if c.isalnum())
        else:
            # Convert image to RGB (OpenCV loads as BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Regular OCR for other fields
            extracted_text = pytesseract.image_to_string(image, lang='eng+rus+kaz')
        
        return extracted_text.strip()
    except Exception as e:
        print(f"Error in text extraction for {filename}: {str(e)}")
        return ""

def save_debug_image(image, filename, suffix):
    """Save intermediate processing results for debugging"""
    debug_dir = "debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    output_path = os.path.join(debug_dir, f"{filename}_{suffix}.jpg")
    cv2.imwrite(output_path, image)

# Directory containing images and output file
image_dir = 'detected_parts'
output_file = 'extracted_text.json'

# Dictionary to hold the extracted text
extracted_texts = {}

# Files to skip
skip_files = {'annotated_full.jpg'}

# Loop through all images in the directory
for filename in os.listdir(image_dir):
    # Skip unwanted files
    if filename in skip_files:
        continue
        
    if filename.endswith(".jpg") or filename.endswith(".png"):
        try:
            image_path = os.path.join(image_dir, filename)
            text = extract_text_from_image(image_path, filename)
            
            # Parse filename for label and confidence
            if '_' in filename:
                parts = filename.split('_')
                label = parts[0]
                # Extract confidence value, removing file extension
                confidence = float(parts[1].split('.')[0])
            else:
                label = filename.replace('.jpg', '').replace('.png', '')
                confidence = 1.0
            
            extracted_texts[label] = {
                'text': text,
                'confidence': confidence
            }
            print(f"Processed {label}: {text}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

# Save the extracted text to a JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted_texts, f, ensure_ascii=False, indent=4)

print(f'Text extracted and saved to {output_file}')
