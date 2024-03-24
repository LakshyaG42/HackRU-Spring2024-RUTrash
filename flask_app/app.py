import os
from flask import Flask, request, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision import models
import torch.nn as nn

app = Flask(__name__)

# Load the saved MobileNetV2 model
model = models.mobilenet_v2()
num_classes = 12
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']  
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.Linear(1024, num_classes)
)
model.load_state_dict(torch.load('garbage_classification_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define transformations for the input image
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return render_template('index.html', error='No file selected')

        # Check if the file format is allowed
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        if not file.filename.lower().endswith(tuple(allowed_extensions)):
            return render_template('index.html', error='Invalid file format')

        # Process the uploaded image
        image = Image.open(file.stream)
        image_tensor = data_transforms(image).unsqueeze(0)  # Add batch dimension

        # Make predictions
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        # Get the predicted class label
        predicted_class = predicted.item()
        predicted_label = class_names[predicted_class]

        return render_template('result.html', image_path=file.filename, predicted_label=predicted_label)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)