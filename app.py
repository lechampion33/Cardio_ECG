from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from ecg import ECG
import os
import matplotlib.pyplot as plt 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'

# Initialize ECG object
ecg = ECG()

def save_image(filename, folder, image):
    """Utility to save images."""
    filepath = os.path.join(folder, filename)
    plt.imsave(filepath, image, cmap='gray')
    return filepath
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['ecg_image']  # Note: Name matches the `name="ecg_image"` in the HTML
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('process_ecg', filename=filename))
    return render_template('upload.html')

@app.route('/cardioinsight')
def cardioinsight():
    return render_template('cardioinsight.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/process/<filename>', methods=['GET'])
def process_ecg(filename):
    # Load the uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_image = ecg.getImage(filepath)

    # Save and display the uploaded image
    uploaded_image_path = save_image('uploaded_image.png', app.config['PROCESSED_FOLDER'], uploaded_image)

    # Grayscale conversion
    gray_image = ecg.GrayImage(uploaded_image)
    gray_image_path = save_image('gray_image.png', app.config['PROCESSED_FOLDER'], gray_image)

    # Divide leads
    leads = ecg.DividingLeads(uploaded_image)
    leads_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'Leads_1-12_figure.png')
    long_lead_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'Long_Lead_13_figure.png')

    # Preprocessed leads
    ecg.PreprocessingLeads(leads)
    preprocessed_leads_path = os.path.join(app.config['PROCESSED_FOLDER'], 'Preprossed_Leads_1-12_figure.png')
    preprocessed_long_lead_path = os.path.join(app.config['PROCESSED_FOLDER'], 'Preprossed_Leads_13_figure.png')

    # Contour leads
    ecg.SignalExtraction_Scaling(leads)
    contour_leads_path = os.path.join(app.config['PROCESSED_FOLDER'], 'Contour_Leads_1-12_figure.png')

    # Combine and perform dimensionality reduction and classification
    combined_csv = ecg.CombineConvert1Dsignal()
    reduced_df = ecg.DimensionalReduction(combined_csv)
    result = ecg.ModelLoad_predict(reduced_df)
    # Prepare data for frontend
    ecg_1d_signal = combined_csv.head(10).to_html(classes='table table-bordered', index=False)
    reduced_data = reduced_df.head(10).to_html(classes='table table-bordered', index=False)

    return render_template(
        'process.html',
        uploaded_image_path=uploaded_image_path,
        gray_image_path=gray_image_path,
        leads_image_path=leads_image_path,
        long_lead_image_path=long_lead_image_path,
        preprocessed_leads_path=preprocessed_leads_path,
        preprocessed_long_lead_path=preprocessed_long_lead_path,
        contour_leads_path=contour_leads_path,
        result=result, 
        ecg_1d_signal=ecg_1d_signal, 
        reduced_data=reduced_data
    )

if __name__ == '__main__':
    app.run(debug=True)
