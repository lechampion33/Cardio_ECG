from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = load_model('ecg_cnn_model.h5')

# Define the prediction function
def predict_ecg(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust target size if needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension
    img_array /= 255.0  # Normalization (if needed based on your model)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    if predicted_class == 0:
        return "Your ECG corresponds to Abnormal Heartbeat."
    elif predicted_class == 1:
        return "Your ECG corresponds to Myocardial Infarction."
    elif predicted_class == 2:
        return "Your ECG is Normal."
    elif predicted_class == 3:
        return "Your ECG corresponds to History of Myocardial Infarction."
    else:
        return "Unknown prediction."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        uploaded_file = request.files['ecg_image']
        if uploaded_file.filename != '':
            img_path = f'./static/{uploaded_file.filename}'
            uploaded_file.save(img_path)

            # Predict the class of the ECG image
            result = predict_ecg(img_path)
            return render_template('index.html', result=result)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
