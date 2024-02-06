import sys
import pandas as pd
from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# Limit GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Force CPU usage
tf.config.set_visible_devices([], 'GPU')
app = Flask(__name__)
model = load_model('xray_classification_model.h5')  # Load the trained model
csv_data = pd.read_csv('Data_Entry_2017_v2020.csv')

# Function to get details for a given image index
def get_image_details(image_index):
    image_details = csv_data[csv_data['Image Index'] == image_index]
    return image_details

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')  # Convert to RGB if the image is in grayscale
    img = img.resize((224, 224))  # Resize to match the input size of the model
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def get_diagnosis(prediction, probability, file_name):
    print(f"Raw Model Prediction: {prediction}, Probability: {probability}")
    try:
        if prediction == "Abnormal":
            # Get additional details from the CSV file based on the file name
            details = get_image_details(file_name)

            # Check if details DataFrame is not empty and 'Finding Labels' column exists
            if not details.empty and 'Finding Labels' in details.columns:
                # Extract potential diseases from the "Finding Labels" column
                diseases = details['Finding Labels'].iloc[0].split('|')

                # Include relevant information in the diagnosis output
                if diseases:
                    diagnosis_output = f'The patient might have {", ".join(diseases)} disease.'
                else:
                    diagnosis_output = 'No specific disease detected. Further investigation may be needed.'
            else:
                diagnosis_output = 'No details found for the given image. Further investigation may be needed.'
        else:
            diagnosis_output = f'No abnormalities detected. It looks normal'
    except ValueError:
        diagnosis_output = 'Invalid prediction value. Please check the model output.'

    return diagnosis_output


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file:
            image_path = "uploaded_image.jpg"  # Save the image temporarily
            file.save(image_path)
            processed_image = preprocess_image(image_path)
            prediction_probability = model.predict(processed_image)[0][0]
            prediction = "Normal" if model.predict(processed_image)[0][0] > 0.5 else "Abnormal"
            diagnosis = get_diagnosis(prediction, prediction_probability, file.filename)
            
            return render_template('index.html', diagnosis=diagnosis)

    return render_template('index.html', diagnosis=None)

if __name__ == '__main__':
    app.run(debug=True)
