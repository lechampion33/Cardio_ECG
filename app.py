from flask import Flask, render_template, request, redirect, url_for, send_file, session
import os
import json
from ecg import predict_ecg  # Import the ML function
from werkzeug.utils import secure_filename  # Add this line

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
        uploaded_file = request.files['ecg_image']
        if uploaded_file.filename != '':
            os.makedirs('./static/uploads', exist_ok=True)
            filename = secure_filename(uploaded_file.filename)
            img_path = os.path.join('./static/uploads', filename)
            uploaded_file.save(img_path)

            details, pdf_path, gray_path, binary_path, graph_path, lead_images = predict_ecg(img_path)
            session['lead_images'] = json.dumps(lead_images)  # Store as JSON
            
            return redirect(url_for('result',
                                   img_filename=filename, 
                                   pdf_filename=os.path.basename(pdf_path), 
                                   prediction=details['prediction'],
                                   confidence=details['confidence'],
                                   heart_rate=details.get('heart_rate', 'N/A'),
                                   pr_interval=details.get('pr_interval', 'N/A'),
                                   qrs_duration=details.get('qrs_duration', 'N/A'),
                                   notes=details['notes']))

    return render_template('upload.html')

@app.route('/result')
def result():
    img_filename = request.args.get('img_filename')
    pdf_filename = request.args.get('pdf_filename')
    lead_images = session.get('lead_images', {})

    
    
    if not img_filename or not pdf_filename:
        return redirect(url_for('upload'))
    
    details = {
        "prediction": request.args.get('prediction', 'N/A'),
        "confidence": request.args.get('confidence', 'N/A'),
        "heart_rate": request.args.get('heart_rate', 'N/A'),
        "pr_interval": request.args.get('pr_interval', 'N/A'),
        "qrs_duration": request.args.get('qrs_duration', 'N/A'),
        "notes": request.args.get('notes', 'N/A')
    }
    return render_template('result.html', img_filename=img_filename, pdf_filename=pdf_filename, 
                          details=details, lead_images=lead_images)

@app.route('/download/<pdf_filename>')
def download_pdf(pdf_filename):
    return send_file(f'./static/processed/{pdf_filename}', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
