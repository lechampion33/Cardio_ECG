from flask import Flask, render_template, request, redirect, url_for, send_file, session
import os
import json
from ecg import predict_ecg  # Import the ML function
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'super_secret_key123'  # Simple string for development

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
        # Get patient details from form
        patient_name = request.form.get('patient_name')
        patient_age = request.form.get('patient_age')
        patient_gender = request.form.get('patient_gender')
        uploaded_file = request.files.get('ecg_image')

        # Check if all required fields are provided
        if not uploaded_file or not patient_name or not patient_age or not patient_gender:
            return render_template('upload.html', error="Please provide all required fields: patient name, age, gender, and ECG image.")

        # Validate age (basic check)
        try:
            patient_age = int(patient_age)
            if patient_age <= 0:
                raise ValueError
        except ValueError:
            return render_template('upload.html', error="Age must be a positive number.")

        # Save the uploaded image
        os.makedirs('./static/uploads', exist_ok=True)
        filename = secure_filename(uploaded_file.filename)
        img_path = os.path.join('./static/uploads', filename)
        uploaded_file.save(img_path)

        # Call predict_ecg with patient details
        details, pdf_path, gray_path, binary_path, graph_path, lead_images = predict_ecg(
            img_path, patient_name, patient_age, patient_gender
        )

        # Check if prediction failed
        if details is None:
            return render_template('upload.html', error="Error processing ECG image.")

        # Store lead_images in session as JSON
        session['lead_images'] = json.dumps(lead_images)

        # Redirect to result page with all details
        return redirect(url_for('result',
                                img_filename=filename,
                                pdf_filename=os.path.basename(pdf_path),
                                prediction=details['prediction'],
                                confidence=details['confidence'],
                                heart_rate=details.get('heart_rate', 'N/A'),
                                pr_interval=details.get('pr_interval', 'N/A'),
                                qrs_duration=details.get('qrs_duration', 'N/A'),
                                notes=details['notes'],
                                patient_name=details['patient_name'],
                                patient_age=details['patient_age'],
                                patient_gender=details['patient_gender']))

    return render_template('upload.html')

@app.route('/result')
def result():
    img_filename = request.args.get('img_filename')
    pdf_filename = request.args.get('pdf_filename')
    lead_images = session.get('lead_images', '{}')  # Default to empty JSON string if not found

    if not img_filename or not pdf_filename:
        return redirect(url_for('upload'))

    # Gather all details from query parameters
    details = {
        "prediction": request.args.get('prediction', 'N/A'),
        "confidence": request.args.get('confidence', 'N/A'),
        "heart_rate": request.args.get('heart_rate', 'N/A'),
        "pr_interval": request.args.get('pr_interval', 'N/A'),
        "qrs_duration": request.args.get('qrs_duration', 'N/A'),
        "notes": request.args.get('notes', 'N/A'),
        "patient_name": request.args.get('patient_name', 'N/A'),
        "patient_age": request.args.get('patient_age', 'N/A'),
        "patient_gender": request.args.get('patient_gender', 'N/A')
    }

    return render_template('result.html', img_filename=img_filename, pdf_filename=pdf_filename,
                          details=details, lead_images=lead_images)

@app.route('/download/<pdf_filename>')
def download_pdf(pdf_filename):
    return send_file(f'./static/processed/{pdf_filename}', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)