from skimage.io import imread
from skimage import color
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from numpy import asarray
from skimage.metrics import structural_similarity
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from natsort import natsorted
import joblib

class ECG:
    def __init__(self):
        self.upload_folder = 'static/uploads'  # Folder to save uploaded images
        self.processed_folder = 'static/processed'  # Folder to save processed images
        self.scaled_files_folder = 'static/scaled_files'  # Folder to save scaled CSV files
        
        for folder in [self.processed_folder, self.scaled_files_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def getImage(self, image):
        return imread(image)

    def GrayImage(self, image):
        image_gray = color.rgb2gray(image)
        image_gray = resize(image_gray, (1572, 2213))
        return image_gray

    def DividingLeads(self, image):
        Leads = [
            image[300:600, 150:643],   # Lead 1
            image[300:600, 646:1135],  # Lead aVR
            image[300:600, 1140:1625], # Lead V1
            image[300:600, 1630:2125], # Lead V4
            image[600:900, 150:643],   # Lead 2
            image[600:900, 646:1135],  # Lead aVL
            image[600:900, 1140:1625], # Lead V2
            image[600:900, 1630:2125], # Lead V5
            image[900:1200, 150:643],  # Lead 3
            image[900:1200, 646:1135], # Lead aVF
            image[900:1200, 1140:1625],# Lead V3
            image[900:1200, 1630:2125],# Lead V6
            image[1250:1480, 150:2125] # Long Lead
        ]
        
        return Leads

    def PreprocessingLeads(self, Leads):
        preprocessed_leads = []

        for lead in Leads:
            grayscale = color.rgb2gray(lead)
            blurred_image = gaussian(grayscale, sigma=1)
            global_thresh = threshold_otsu(blurred_image)
            binary_global = blurred_image < global_thresh
            binary_global = resize(binary_global, (300, 450))
            preprocessed_leads.append(binary_global)

        return preprocessed_leads

    def SignalExtraction_Scaling(self, Leads):
        # Clear any existing scaled files
        for file in os.listdir(self.scaled_files_folder):
            os.remove(os.path.join(self.scaled_files_folder, file))

        for idx, lead in enumerate(Leads[:-1]):
            grayscale = color.rgb2gray(lead)
            blurred_image = gaussian(grayscale, sigma=0.7)
            global_thresh = threshold_otsu(blurred_image)
            binary_global = blurred_image < global_thresh
            binary_global = resize(binary_global, (300, 450))
            
            contours = measure.find_contours(binary_global, 0.8)
            largest_contour = max(contours, key=lambda x: x.shape[0])
            resized_contour = resize(largest_contour, (255, 2))

            # Scale the data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(resized_contour)

            # Save scaled data to a CSV file
            scaled_file_name = os.path.join(self.scaled_files_folder, f"Scaled_1DLead_{idx + 1}.csv")
            pd.DataFrame(scaled_data[:, 0], columns=['X']).T.to_csv(scaled_file_name, index=False)

    def CombineConvert1Dsignal(self):
        all_data = []

        for file in natsorted(os.listdir(self.scaled_files_folder)):
            if file.endswith('.csv'):
                data = pd.read_csv(os.path.join(self.scaled_files_folder, file))
                all_data.append(data)

        return pd.concat(all_data, axis=1, ignore_index=True)

    def DimensionalReduction(self, test_final):
        pca_loaded_model = joblib.load('PCA_ECG.pkl')
        reduced_data = pca_loaded_model.transform(test_final)
        return pd.DataFrame(reduced_data)

    def ModelLoad_predict(self, final_df):
        loaded_model = joblib.load('Heart_Disease_Prediction_using_ECG.pkl')
        result = loaded_model.predict(final_df)

        if result[0] == 1:
            return "Your ECG corresponds to Myocardial Infarction."
        elif result[0] == 0:
            return "Your ECG corresponds to Abnormal Heartbeat."
        elif result[0] == 2:
            return "Your ECG is Normal."
        else:
            return "Your ECG corresponds to History of Myocardial Infarction."

# Changes to Frontend
# 1. Ensure the backend call resets the state on the server before uploading a new image.
# 2. Add an endpoint to clear intermediate files if needed.
# 3. Test the updated backend thoroughly to ensure it handles multiple uploads consistently.
