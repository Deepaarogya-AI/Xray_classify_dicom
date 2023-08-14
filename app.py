


import os
from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.models import resnet50, densenet161, resnet101
from PIL import Image
import numpy as np
#from torchvision.models import convnext_small 
from torchvision.models import efficientnet_b4,resnet34
from torchvision.models import resnet50,densenet161,inception_v3,regnet_y_8gf
from torchvision.models import efficientnet_b6
import mritopng
import shutil

from werkzeug.utils import secure_filename
from PIL import UnidentifiedImageError

from flask import Flask, render_template, request, send_file  # Add send_file import
# Other import statements are here...

import io
import csv
import base64
from io import BytesIO
import tempfile

app = Flask(__name__)



UPLOAD_FOLDER = 'uploaded_images'  # Directory to save uploaded images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the ensemble model
num_classes = 2  # Number of classes (e.g., binary classification)

# Define models
def create_resnet_model():
    model = resnet50(pretrained=False)  # Do not use pre-trained weights
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    model.eval()
    return model

def create_densenet_model():
    model = densenet161(pretrained=True)  # Do not use pre-trained weights
    num_features = 2208
    model.classifier = torch.nn.Linear(num_features, num_classes)
    model.eval()
    return model



def create_resnet101_model():
    model = resnet101(pretrained=False)  # Do not use pre-trained weights
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    model.eval()
    return model



def create_resnet34_model():
    model = resnet34(pretrained=False)  # Do not use pre-trained weights
    num_features = model.fc.in_features
    #model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    model.eval()
    return model

resnet_model = create_resnet_model()
resnet101_model = create_resnet101_model()
densenet_model = create_densenet_model()
resnet34_model=create_resnet34_model()




import torch
import torch.nn as nn
from torchvision.models import densenet161

def create_densenet_model(num_classes=2, dropout_rate=0.2):
    model = densenet161(pretrained=False)  # Load the DenseNet model without pretrained weights
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, num_classes)
    )
    model.eval()  # Set the model to evaluation mode
    return model

# Create the model
model_dens = create_densenet_model()

# Load the pretrained weights for matching layers
pretrained_state_dict = torch.load('v2_densenet121_95acc.pth', map_location=torch.device('cpu'))
print("densenet161 model loaded")
model_dict = model_dens.state_dict()
pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model_dens.load_state_dict(model_dict)




def create_densenet_aug_model(num_classes=2, dropout_rate=0.2):
    model = densenet161(pretrained=False)  # Load the DenseNet model without pretrained weights
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, num_classes)
    )
    model.eval()  # Set the model to evaluation mode
    return model

# Create the model
model_dens_aug = create_densenet_aug_model()

# Load the pretrained weights for matching layers
pretrained_state_dict = torch.load('v2_densnet161_drop_dataaug.pth', map_location=torch.device('cpu'))
print("densenet161_aug model loaded")
model_dict = model_dens_aug.state_dict()
pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model_dens_aug.load_state_dict(model_dict)



# Load the individual models' weights
resnet_model.load_state_dict(torch.load('v2_resnet50new_adamW_withcrossenloss94acc.pth', map_location=torch.device('cpu')))
#resnet34_model.load_state_dict(torch.load('.pth', map_location=torch.device('cpu')))
resnet101_model.load_state_dict(torch.load('v2_resnet101_heavydataaug_pretrain.pth', map_location=torch.device('cpu')))
#densenet_model.load_state_dict(torch.load('v2_densenet121_95acc.pth',map_location=torch.device('cpu')))
    #best_model_DenseNet.pth', map_location=torch.device('cpu')))



# Define a function to preprocess the input image
def preprocess_image(image, model_name, image_size):
    try:

        image_tensor = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])(image)

        # Expand the single-channel image tensor to 3 channels
        image_tensor = image_tensor.expand(3, -1, -1)

        # Normalize the image using the standard ImageNet normalization values
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)

        return image_tensor.unsqueeze(0)
    except UnidentifiedImageError as e:
        print(f"Error processing image: {e}")
        return None


# Define a function for ensemble inference
def ensemble_inference(models, image, image_size):
    image_tensor = preprocess_image(image, models[0].__class__.__name__, image_size)

    with torch.no_grad():
        predictions = []
        for model in models:
            outputs = model(image_tensor)
            predictions.append(F.softmax(outputs, dim=1).squeeze().numpy())

        predictions = np.mean(predictions, axis=0)
        return np.argmax(predictions), predictions


def is_valid_dicom(dicom_file):
    try:
        ds = pydicom.dcmread(dicom_file)
        return True
    except Exception as e:
        print(f"Invalid DICOM file: {e}")
        return False



def dicom_to_jpg(dicom_file):
    try:
        # Read the DICOM file
        ds = pydicom.dcmread(dicom_file)

        # Get the pixel array from DICOM
        pixel_array = ds.pixel_array

        # Convert the pixel array to an image
        image = Image.fromarray(pixel_array)

        # Convert the image to RGB mode (for JPEG conversion)
        image = image.convert('RGB')

        return image

    except Exception as e:
        print(f"Error converting DICOM to JPEG: {str(e)}")
        return None


CONVERTED_IMAGES_FOLDER = 'converted_images'

if not os.path.exists(CONVERTED_IMAGES_FOLDER):
    os.makedirs(CONVERTED_IMAGES_FOLDER)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        images = request.files.getlist('images')  # Get the list of uploaded images
        if not images:
            return render_template('index.html', message='No files selected')

        predictions_data = []
        for image in images:
            image_filename = secure_filename(image.filename)
            image_format = image_filename.split('.')[-1].lower()
            if image_format == 'dcm':
                # Convert DICOM to PNG using mritopng
                dicom_path = os.path.join(CONVERTED_IMAGES_FOLDER, image_filename)
                image.save(dicom_path)
                png_path = os.path.join(CONVERTED_IMAGES_FOLDER, f"{image_filename}.png")
                mritopng.convert_file(dicom_path, png_path, auto_contrast=True)
                png_image = Image.open(png_path).convert('RGB')
                image_for_prediction = png_image

                # Read the PNG image and convert to base64-encoded data
                with open(png_path, "rb") as image_file:
                    base64_image_data = base64.b64encode(image_file.read()).decode("utf-8")

        
                # Define the label map
                label_map = {0: 'healthy', 1: 'unhealthy'}

                threshold = 0.50
                # Perform ensemble inference with image size (224, 224) for example
                image_size = (224, 224)  # You can choose a different size as needed

                # Perform ensemble inference and get predictions as before
                predicted_label_index, predictions = ensemble_inference([resnet_model, model_dens, model_dens_aug], image_for_prediction, image_size)
                confidence_score = predictions[predicted_label_index]

                # Check if confidence score is above the threshold
                if confidence_score > threshold:
                    predicted_label_str = label_map[predicted_label_index]
                else:
                    predicted_label_str = 'No result'

                # Store prediction results in a dictionary
                prediction_data = {
                    "image_name": image_filename,
                    "predicted_label": predicted_label_str,
                    "confidence_score": confidence_score,
                    "base64_image_data": base64_image_data  # Store the Base64-encoded image data
                }

                predictions_data.append(prediction_data)


        # Save the results to a CSV file
        save_to_csv(predictions_data)

        # Redirect to the prediction results page
        return render_template('prediction_result.html', predictions=predictions_data)

    return render_template('index.html')


@app.route('/convert_dicom_to_jpg', methods=['GET', 'POST'])
def convert_dicom_to_jpg():
    if request.method == 'POST':
        try:
            input_folder = request.form.get('input_folder')  # Get the selected input folder
            output_folder = r'outputdir'
            
            # Delete the existing output folder if it already exists
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            
            # Convert a whole folder recursively
            mritopng.convert_folder(input_folder, output_folder)
            
            message = "DICOM to JPG conversion successful"
        except Exception as e:
            message = f"Error during DICOM to JPG conversion: {str(e)}"
        
        return render_template('index.html', dicom_message=message)

    return render_template('index.html')


@app.route('/download_csv')
def download_csv():
    try:
        # Send the CSV file to the user as an attachment
        return send_file('prediction_results.csv',
                         mimetype='text/csv',
                         attachment_filename='prediction_results.csv',
                         as_attachment=True)
    except Exception as e:
        return render_template('error.html', error=str(e))



# Function to save prediction results to CSV file
def save_to_csv(predictions_data):
    with open('prediction_results.csv', mode='w', newline='') as csv_file:
        fieldnames = ['Image Name', 'Predicted Label', 'Confidence Score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for prediction in predictions_data:
            writer.writerow({
                'Image Name': prediction['image_name'],
                'Predicted Label': prediction['predicted_label'],
                'Confidence Score': prediction['confidence_score'],
            })

if __name__ == '__main__':
    app.run(debug=True)




# Use an official Python runtime as a parent image
# FROM python:3.8-slim

# # Set the working directory to /app
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . /app

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Expose port 8080 for the Flask app
# EXPOSE 8080

# # Run the Flask app
# CMD ["python", "app.py"]


# docker build -t flask-app .

# docker run -p 8080:8080 flask-app


#sudo docker run -it app_classify /bin/bash
