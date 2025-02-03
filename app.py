from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
import numpy as np
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.transform import resize
import os

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = load_model('ecg_cnn_model.h5')

# Define the prediction function
def preprocess_custom_image(image_path):
    # Load and preprocess EXACTLY like training data
    image = imread(image_path)
    
    # Convert to grayscale
    if len(image.shape) == 3:  # Color image
        gray_image = rgb2gray(image)
    else:  # Already grayscale
        gray_image = image
    
    # Apply preprocessing pipeline (same as during training)
    blurred_image = gaussian(gray_image, sigma=0.7)
    otsu_threshold = threshold_otsu(blurred_image)
    binary_image = blurred_image < otsu_threshold  # Important: Same thresholding direction!
    resized_image = resize(binary_image, (128, 128))
    
    # Add channel dimension and batch dimension
    processed_image = resized_image.reshape(1, 128, 128, 1)  # (batch, height, width, channels)
    return processed_image

def predict_ecg(img_path):
    # Preprocess the image for prediction
    processed_image = preprocess_custom_image(img_path)
    
    # Predict the class
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    if predicted_class == 0:
        return "Your ECG corresponds to Abnormal Heartbeat."
    elif predicted_class == 1:
        return "Your ECG corresponds to History of Myocardial Infarction."
    elif predicted_class == 2:
        return "Your ECG corresponds to Myocardial Infarction."
    elif predicted_class == 3:
        return "Your ECG corresponds to Normal."
    else:
        return "Unknown prediction."

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/cardioinsight')
def cardioinsight():
    return render_template('cardioinsight.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Handle file upload
        uploaded_file = request.files['ecg_image']
        if uploaded_file.filename != '':
            img_path = f'./static/{uploaded_file.filename}'
            uploaded_file.save(img_path)

            # Predict the class of the ECG image
            result = predict_ecg(img_path)
            return redirect(url_for('result', prediction=result))

    return render_template('upload.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True)