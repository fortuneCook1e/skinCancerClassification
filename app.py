from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import tempfile
import base64

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['SEGMENTED_FOLDER'] = 'segmented_images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

def get_segment(model, image):
    # Predict the mask
    pred_mask = model.predict(np.expand_dims(image, axis=0))[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Apply threshold
    # Find contours
    contours_pred, _ = cv2.findContours(pred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # The image is already in RGB format suitable for Matplotlib, no need to convert
    original_image_vis = image.copy()
    # Convert image to BGR for OpenCV if needed
    original_image_vis_cv2 = (original_image_vis * 255).astype(np.uint8)
    original_image_vis_cv2 = cv2.cvtColor(original_image_vis_cv2, cv2.COLOR_RGB2BGR)
    # Draw contours on the original image for OpenCV
    cv2.drawContours(original_image_vis_cv2, contours_pred, -1, (255, 0, 0), 3)  # Predicted mask in blue
    return original_image_vis_cv2, pred_mask

# def preprocess_image(image_path, target_size=(192, 256)):
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, target_size) / 255.0  # Normalize to 0-1
#     return image

def preprocess_image(image_path, target_size=(192, 256)):
    image = cv2.imread(image_path)
    original_size = image.shape[:2]  # Capture original size before resizing
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, target_size) / 255.0  # Normalize to 0-1
    return image, original_size  # Return both processed image and original size

def resize_image_to_original(img_with_contour, original_size):
    # Resize back to original size
    return cv2.resize(img_with_contour, (original_size[1], original_size[0]))  # Width and height are reversed in cv2


def get_region(original_image, pred_mask):
    original_image = (original_image * 255).astype(np.uint8)
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)  # if open: 27
    # Use the binary mask to extract the segmented region
    segmented_region = cv2.bitwise_and(original_image, original_image, mask=pred_mask)
    return segmented_region

def calculate_confidence(prediction):
    # Calculate the distance from 0.5
    distance_from_0_5 = abs(0.5 - prediction)
    # Scale the confidence such that 0.5 -> 0% and 1 (or 0) -> 100%
    confidence = distance_from_0_5 * 2
    return confidence

def classify_img(model, segmented_region):
    # Preprocess the segmented region to match the classification model input
    # Resize the segmented part to match the classification model's expected input size
    segmented_resized = cv2.resize(segmented_region, (192, 256))
    # Normalize the segmented part if necessary (depends on your model's training process)
    segmented_resized = segmented_resized / 255.0
    # Add batch dimension and predict the class using the classification model
    classification_pred = model.predict(np.expand_dims(segmented_resized, axis=0))
    # The output 'classification_pred' is the probability of the positive class
    prediction_score = classification_pred[0][0]
    confidence_score = calculate_confidence(prediction_score)
    predicted_class_label = "Malignant" if prediction_score > 0.5 else "Benign"
    return predicted_class_label, confidence_score

# Load your models here (assuming they are loaded only once to save resources)
segmentation_model = load_model('C:/Users/jeesh/Documents(local)/FYP/fyp code/chosen_models/resnet50_segmentation-unfreeze.h5')
classification_model = load_model('C:/Users/jeesh/Documents(local)/FYP/fyp code/chosen_models/AttentionInceptionV3(with_extraction)-r8_k3.h5')

# Get a list of all image files in the source directory
# source_directory = "C:/Users/jeesh/Documents(local)/FYP/test set/splitted_benign_malignant_testset(original)/1"

# image_files = [f for f in os.listdir(source_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# counter = 0
# # Loop through all images with a progress bar
# for image_file in image_files:
#     image_path = os.path.join(source_directory, image_file)
#     preprocessed_image, original_size = preprocess_image(image_path)
#     # Get the contour and mask
#     img_with_contour, pred_mask = get_segment(segmentation_model, preprocessed_image)
#     # Get tumour region
#     segmented_region = get_region(preprocessed_image, pred_mask)

#     # Perform classification on the segmented region
#     predicted_class_label, confidence_score = classify_img(classification_model, segmented_region)
#     if (predicted_class_label == 'Malignant'):
#         print(image_file)
#         counter = counter + 1
# print("All images have been processed and saved.")
# print("malignant count = ", counter)

# # Ensure the upload and segmented folders exist
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs(app.config['SEGMENTED_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/contact')
def contact():
    # Process form data here
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the uploaded file to a temporary directory
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.save(temp.name)
            # Preprocess the uploaded image
            preprocessed_image, original_size = preprocess_image(temp.name)
            # Get the contour and mask
            img_with_contour, pred_mask = get_segment(segmentation_model, preprocessed_image)
            # Get tumour region
            segmented_region = get_region(preprocessed_image, pred_mask)

        # Perform classification on the segmented region
        predicted_class_label, confidence_score = classify_img(classification_model, segmented_region)
        
        # Resize the image with contour back to original size
        img_with_contour_resized = resize_image_to_original(img_with_contour, original_size)
        # segmented_region_resized = resize_image_to_original(segmented_region, original_size)
        # display_segmented_region = cv2.cvtColor(segmented_region_resized, cv2.COLOR_RGB2BGR)
        pred_mask_resized = resize_image_to_original(pred_mask, original_size)

        # Encode the segmented image to base64 to send in the JSON response
        _, buffer = cv2.imencode('.png', img_with_contour_resized)
        b64_segmented_image = base64.b64encode(buffer).decode('utf-8')
        
        # _, buffer2 = cv2.imencode('.png', display_segmented_region)
        # b64_mask_image = base64.b64encode(buffer2).decode('utf-8')
        
         # Encode pred_mask to base64
        _, buffer3 = cv2.imencode('.png', pred_mask_resized * 255)  # Convert binary mask to a visible format
        b64_mask = base64.b64encode(buffer3).decode('utf-8')

        # Return the classification result and the base64-encoded image
        return jsonify({
            'segmented_image': f"data:image/png;base64,{b64_segmented_image}",
            'mask_image': f"data:image/png;base64,{b64_mask}",
            'predicted_class': predicted_class_label,
            'confidence_score': f"{confidence_score:.2f}"
        })


# @app.route('/segmented_images/<filename>')
# def send_segmented_file(filename):
#     return send_from_directory(app.config['SEGMENTED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
