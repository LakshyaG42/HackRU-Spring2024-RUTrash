import os
from flask import Flask, jsonify, request, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision import models
import torch.nn as nn
import requests
from markdown import markdown
app = Flask(__name__)

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

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No file selected')

        allowed_extensions = {'jpg', 'jpeg', 'png'}
        if not file.filename.lower().endswith(tuple(allowed_extensions)):
            return render_template('index.html', error='Invalid file format')

        image = Image.open(file.stream)
        image_tensor = data_transforms(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        predicted_class = predicted.item()
        predicted_label = class_names[predicted_class]

        return render_template('result.html', image_path=file.filename, predicted_label=predicted_label)

    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate_carbon_footprint():
    electric_bill = float(request.form['electric_bill'])
    gas_bill = float(request.form['gas_bill'])
    oil_bill = float(request.form['oil_bill'])
    car_mileage = float(request.form['car_mileage'])
    flights_short = int(request.form['flights_short'])
    flights_long = int(request.form['flights_long'])
    recycle_newspaper = request.form.get('recycle_newspaper') == 'on'
    recycle_aluminum = request.form.get('recycle_aluminum') == 'on'

    carbon_footprint = (
        electric_bill * 105 +
        gas_bill * 105 +
        oil_bill * 113 +
        car_mileage * 0.79 +
        flights_short * 1100 +
        flights_long * 4400 +
        (184 if not recycle_newspaper else 0) +
        (166 if not recycle_aluminum else 0)
    )

    return render_template('carbon_footprint_result.html', carbon_footprint=carbon_footprint)

CLOUDFLARE_API_TOKEN = os.getenv('CLOUDFLARE_API_TOKEN')
CLOUDFLARE_ACCOUNT_ID = os.getenv('CLOUDFLARE_ACCOUNT_ID')
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        model = "@cf/meta/llama-2-7b-chat-int8"
        query = request.form['query']

        response = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{model}",
            headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
            json={"messages": [
                {"role": "user", "content": query}
            ]}
        )

        inference = response.json()
        markdown_response = inference.get("result", {}).get("response", "")

        return render_template('climateExpert.html', response=markdown(markdown_response), input_query=query)

    return render_template('climateExpert.html')


if __name__ == '__main__':
    app.run(debug=True)
