import os
import cv2
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.transform import resize
from skimage import img_as_ubyte
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader

# Constants
PROCESSED_FOLDER = './static/processed'
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the trained model and LabelEncoder
try:
    model = load_model('ecg_cnn_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    class_labels = encoder.classes_  # ["Abnormal", "History_MI", "MI", "Normal"]
    # Convert to list to avoid NumPy array issues
    class_labels = list(class_labels) if class_labels is not None else []
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    class_labels = []

def preprocess_custom_image(image_path):
    image = imread(image_path)
    if len(image.shape) == 3:
        gray_image = rgb2gray(image)
    else:
        gray_image = image
    
    blurred_image = gaussian(gray_image, sigma=0.7)
    otsu_threshold = threshold_otsu(blurred_image)
    binary_image = blurred_image < otsu_threshold
    resized_image = resize(binary_image, (128, 128))
    processed_image = resized_image.reshape(1, 128, 128, 1)
    
    gray_path = os.path.join(PROCESSED_FOLDER, f'gray_{os.path.basename(image_path)}')
    binary_path = os.path.join(PROCESSED_FOLDER, f'binary_{os.path.basename(image_path)}')
    
    imsave(gray_path, img_as_ubyte(gray_image))
    imsave(binary_path, img_as_ubyte(binary_image))
    
    return processed_image, gray_image, binary_image, gray_path, binary_path

def estimate_signal_quality(binary_image):
    edge_density = np.mean(binary_image)
    if edge_density < 0.2:
        return "Poor (low contrast)"
    elif edge_density < 0.5:
        return "Average"
    else:
        return "Good"

def detect_artifacts(binary_image):
    noise_level = np.var(binary_image)
    return "Artifacts detected" if noise_level > 0.02 else "No significant artifacts"

def split_ecg_leads(image_path):
    """Split a 12-lead ECG image into individual leads and apply preprocessing steps."""
    image = imread(image_path)
    print(f"Image loaded from {image_path} with shape: {image.shape}")
    
    expected_height, expected_width = 1572, 2213  # Expected reference size

    # Calculate scaling factors
    height_scale = image.shape[0] / expected_height
    width_scale = image.shape[1] / expected_width
    
    print(f"Original image shape: {image.shape}")
    print(f"Height scale: {height_scale}, Width scale: {width_scale}")

    # Define and scale coordinates
    lead_coords = {
        "I": (300, 150, 600, 643),
        "aVR": (300, 646, 600, 1135),
        "V1": (300, 1140, 600, 1625),
        "V4": (300, 1630, 600, 2125),
        "II": (600, 150, 900, 643),
        "aVL": (600, 646, 900, 1135),
        "V2": (600, 1140, 900, 1625),
        "V5": (600, 1630, 900, 2125),
        "III": (900, 150, 1200, 643),
        "aVF": (900, 646, 1200, 1135),
        "V3": (900, 1140, 1200, 1625),
        "V6": (900, 1630, 1200, 2125),
    }
    
    scaled_coords = {
        lead: (
            int(y_start * height_scale),
            int(x_start * width_scale),
            int(y_end * height_scale),
            int(x_end * width_scale)
        )
        for lead, (y_start, x_start, y_end, x_end) in lead_coords.items()
    }
    
    lead_images = {}
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    for lead, (y_start, x_start, y_end, x_end) in scaled_coords.items():
        print(f"{lead}: Cropping at [{y_start}:{y_end}, {x_start}:{x_end}]")
        
        # Ensure values are within image bounds
        y_start = max(0, min(y_start, image.shape[0] - 1))
        y_end = max(y_start + 1, min(y_end, image.shape[0]))
        x_start = max(0, min(x_start, image.shape[1] - 1))
        x_end = max(x_start + 1, min(x_end, image.shape[1]))

        lead_image = image[y_start:y_end, x_start:x_end]
        print(f"Cropped {lead} at [{y_start}:{y_end}, {x_start}:{x_end}] with shape: {lead_image.shape}")

        if lead_image.size == 0 or lead_image.shape[0] == 0 or lead_image.shape[1] == 0:
            print(f"Skipping empty lead image for {lead}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(lead_image, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Otsu's Thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #invert
        inverted = cv2.bitwise_not(binary)

        # Save only the final processed binary image
        lead_path = os.path.join("static/processed", f"{base_filename}_lead_{lead}.png")
        imsave(lead_path, inverted)
        print(f"Saved final processed {lead} lead at {lead_path}")

        lead_images[lead] = f"processed/{base_filename}_lead_{lead}.png"

    return lead_images

def predict_ecg(img_path):
    if len(class_labels) == 0:
        return None, None, None, None, None, None
    
    lead_images = split_ecg_leads(img_path)
    
    processed_image, gray_image, binary_image, gray_path, binary_path = preprocess_custom_image(img_path)
    predictions = model.predict(processed_image)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class_idx]
    confidence = np.max(predictions) * 100

    # Map abbreviated labels to full names
    label_mapping = {
        "Normal": "Normal",
        "Abnormal": "Abnormal Rhythm",
        "MI": "Myocardial Infarction",
        "History_MI": "History of Myocardial Infarction"
    }
    display_label = label_mapping.get(predicted_label, predicted_label)  # Fallback to original if not in map

    signal_quality = estimate_signal_quality(binary_image)
    artifact_status = detect_artifacts(binary_image)

    details = {
        "prediction": display_label,  # Use full name here
        "confidence": f"{confidence:.2f}%",
        "signal_quality": signal_quality,
        "artifact_status": artifact_status,
        "heart_rate": "N/A",
        "pr_interval": "N/A",
        "qrs_duration": "N/A",
        "notes": get_clinical_notes(predicted_label)  # Keep original label for notes
    }

    graph_path = os.path.join(PROCESSED_FOLDER, f'graph_{os.path.basename(img_path)}.png')
    plt.figure(figsize=(5, 3))
    plt.imshow(binary_image, cmap='gray')
    plt.title(f"Preprocessed ECG ({display_label})")
    plt.axis('off')
    plt.savefig(graph_path)
    plt.close()

    pdf_path = generate_pdf(details, img_path, binary_path, lead_images)
    
    return details, pdf_path, gray_path, binary_path, graph_path, lead_images

def get_clinical_notes(label):
    notes = {
        "Normal": (
            "No abnormalities detected. The ECG shows a regular heart rhythm, "
            "normal P-wave morphology, and appropriate QRS and T-wave formations. "
            "Routine monitoring is advised. Maintain a healthy lifestyle to prevent future risks."
        ),
        "Abnormal": (
            "Potential arrhythmia detected. This could indicate irregular heartbeats such as atrial fibrillation, "
            "ventricular tachycardia, or other conduction abnormalities. A cardiology consultation is recommended for further evaluation, "
            "including Holter monitoring or an echocardiogram if needed."
        ),
        "MI": (
            "Signs of Myocardial Infarction (Heart Attack) detected. "
            "The ECG shows ST-segment elevation or pathological Q waves, indicating possible ischemic damage. "
            "Urgent medical attention is advised. Immediate intervention, such as thrombolytic therapy or PCI, may be required."
        ),
        "History_MI": (
            "Evidence of a previous Myocardial Infarction (Heart Attack) detected. "
            "This may include Q waves, T-wave inversions, or abnormal R-wave progression. "
            "Continuous monitoring, lifestyle modifications, and adherence to prescribed medications (e.g., beta-blockers, statins) "
            "are essential to prevent recurrence."
        )
    }

    return notes.get(label, "No information available for this condition.")


from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
import os

def generate_pdf(details, img_path, preprocessed_img_path, lead_images):
    pdf_path = os.path.join(PROCESSED_FOLDER, f'ecg_report_{os.path.basename(img_path)}.pdf')
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4  # 595 x 842 points

    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.leading = 12

    # --- Page 1: Heading and Images ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "ECG Analysis Report")

    y = height - 80
    c.setFont("Helvetica", 12)
    c.drawString(50, y, "Uploaded ECG Image:")
    y -= 20

    max_img_width = width - 100
    img_reader = ImageReader(img_path)
    img_width, img_height = img_reader.getSize()
    aspect_ratio = img_height / img_width
    img_width_points = max_img_width
    img_height_points = img_width_points * aspect_ratio

    available_height = y - 50
    if img_height_points > available_height:
        img_height_points = available_height
        img_width_points = img_height_points / aspect_ratio

    c.drawImage(img_path, 50, y - img_height_points, width=img_width_points, height=img_height_points, preserveAspectRatio=True)
    y -= (img_height_points + 30)

    c.drawString(50, y, "Preprocessed Binary Image:")
    y -= 20

    if os.path.exists(preprocessed_img_path):
        img_reader = ImageReader(preprocessed_img_path)
        img_width, img_height = img_reader.getSize()
        aspect_ratio = img_height / img_width
        img_width_points = max_img_width
        img_height_points = img_width_points * aspect_ratio

        available_height = y - 50
        if img_height_points > available_height:
            img_height_points = available_height
            img_width_points = img_height_points / aspect_ratio

        c.drawImage(preprocessed_img_path, 50, y - img_height_points, width=img_width_points, height=img_height_points, preserveAspectRatio=True)
    else:
        c.drawString(50, y - 20, "Preprocessed image not found at: " + preprocessed_img_path)

    # --- Page 2: Text Details with Explanation ---
    c.showPage()
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "ECG Analysis Details")
    y -= 30

    available_width = width - 100
    for key, value in details.items():
        text = f"<b>{key.replace('_', ' ').title()}:</b> {value}"
        p = Paragraph(text, normal_style)
        p_width, p_height = p.wrap(available_width, height - y)
        if y - p_height < 50:
            c.showPage()
            y = height - 50
        p.drawOn(c, 50, y - p_height)
        y -= p_height + 10

    # Add explanation for "N/A" values
    explanation = (
        "<b>Note:</b> Heart Rate, PR Interval, and QRS Duration are listed as 'N/A' because these measurements require precise temporal data from the raw ECG signal, "
        "which cannot be accurately extracted from a static image alone. This analysis uses a preprocessed image for classification (e.g., Normal, MI) rather than "
        "detailed waveform timing. To obtain these values, raw signal data (e.g., from an ECG device in digital format) or advanced image digitization techniques "
        "would be needed."
    )
    p = Paragraph(explanation, normal_style)
    p_width, p_height = p.wrap(available_width, height - y)
    if y - p_height < 50:
        c.showPage()
        y = height - 50
    p.drawOn(c, 50, y - p_height)
    y -= p_height + 10

    # --- Page 3: ECG Leads with Descriptions ---
    c.showPage()
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "ECG Lead Images (2x6)")
    y -= 30

    lead_descriptions = {
        "I": "Measures from right arm (-) to left arm (+). Views the lateral wall of the left ventricle. Detects lateral infarction (e.g., LAD artery).",
        "II": "Measures from right arm (-) to left leg (+). Views the inferior wall diagonally. Ideal for rhythm analysis and inferior infarction (e.g., RCA).",
        "III": "Measures from left arm (-) to left leg (+). Views the inferior wall vertically. Confirms inferior infarction with Lead II.",
        "aVR": "Right arm (+). Views the upper right heart. Shows reciprocal changes; useful for right-sided issues or lead misplacement.",
        "aVL": "Left arm (+). Views the high lateral wall. Detects lateral infarction (e.g., circumflex artery) with Lead I.",
        "aVF": "Left leg (+). Views the inferior wall vertically. Key for inferior infarction (e.g., RCA) with Leads II and III.",
        "V1": "4th intercostal space, right sternal border. Views the right ventricle and septum. Detects RV hypertrophy or posterior infarction.",
        "V2": "4th intercostal space, left sternal border. Views the septum and anterior wall. Indicates anterior infarction (e.g., LAD).",
        "V3": "Midway between V2 and V4. Views the anterior wall. Part of anterior lead group for infarction diagnosis.",
        "V4": "5th intercostal space, midclavicular line. Views the anterior wall and apex. Monitors apical involvement.",
        "V5": "5th intercostal space, anterior axillary line. Views the low lateral wall. Detects lateral infarction with V6.",
        "V6": "5th intercostal space, midaxillary line. Views the mid-lateral wall. Complements V5 for lateral wall assessment."
    }

    cols = 2
    rows = 6
    available_width = width - 100
    lead_width = (available_width - (cols - 1) * 10) / cols
    lead_height = lead_width * 0.5
    text_width = lead_width
    x_start = 50
    y_start = y
    x_gap = lead_width + 10
    y_gap = lead_height + 60

    lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    row = 0
    col = 0

    c.setFont("Helvetica", 10)
    for lead in lead_order:
        if lead in lead_images:
            lead_img_path = os.path.join('static', lead_images[lead]).replace('\\', '/')
            x_pos = x_start + (col * x_gap)
            y_pos = y_start - (row * y_gap)

            if y_pos - (lead_height + 40) < 50:
                c.showPage()
                y_start = height - 50
                c.setFont("Helvetica-Bold", 16)
                c.drawString(50, y_start, "ECG Lead Images (Continued)")
                y_start -= 30
                c.setFont("Helvetica", 10)
                row = 0
                y_pos = y_start - (row * y_gap)

            c.drawString(x_pos, y_pos + 10, f"Lead {lead}")
            c.drawImage(
                lead_img_path,
                x_pos, y_pos - lead_height,
                width=lead_width,
                height=lead_height,
                preserveAspectRatio=True
            )

            desc_text = lead_descriptions.get(lead, "No description available.")
            p = Paragraph(desc_text, normal_style)
            p_width, p_height = p.wrap(text_width, 50)
            p.drawOn(c, x_pos, y_pos - lead_height - p_height - 5)

            col += 1
            if col == cols:
                col = 0
                row += 1

    c.showPage()
    c.save()
    return pdf_path