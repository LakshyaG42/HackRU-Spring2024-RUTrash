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

recommendations = {
    'battery' : "Non-rechargeable and alkaline household batteries may be discarded with regular trash. If you need to get rid of rechargeable batteries, you can: \n Drop them off at a Special Waste Disposal site \n Bring them to a store that sells rechargeable batteries or products containing them",
    'biological' : "Biological waste, such as food scraps or organic materials, can be composted if possible. Otherwise put it in the trash!",
    'brown-glass': "Glass containers, such as bottles or jars, can be recycled in glass recycling bins. \n Make sure to rinse the containers before recycling to remove any residue.",
    'green-glass': "Glass containers, such as bottles or jars, can be recycled in glass recycling bins. \n Make sure to rinse the containers before recycling to remove any residue.",
    'white-glass': "Glass containers, such as bottles or jars, can be recycled in glass recycling bins. \n Make sure to rinse the containers before recycling to remove any residue.",
    'cardboard': "Cardboard boxes and packaging materials should be flattened and recycled in designated cardboard recycling bins.",
    'clothes' : "Clothes in good condition can be donated to charity organizations or thrift stores for reuse. Worn-out or damaged clothes can be recycled at textile recycling centers or repurposed into cleaning rags.",
    'metal' : 'Metal items, such as aluminum cans, steel containers, or small metal objects, can be recycled in metal recycling bins. \n Check with local recycling centers for specific guidelines on metal recycling.',
    'paper': "Paper materials, including newspapers, magazines, office paper, and cardboard packaging, can be recycled in paper recycling bins.",
    'plastic' : 'Plastic containers, bottles, and packaging materials should be recycled in plastic recycling bins.',
    'shoes' : 'Donate gently used shoes to charitable organizations or shoe donation centers for reuse. \n Worn-out or damaged shoes may be recyclable at certain recycling facilities that accept textiles and footwear.',
    'trash' : "Items that cannot be recycled or reused should be disposed of in regular household waste bins."
}

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
        recommendation = recommendations[predicted_label]
        isRecycle = predicted_label in ['paper', 'plastic', 'green-glass', 'brown-glass', 'white-glass', 'cardboard', 'metal']
        isTrash = predicted_label in ['biological', 'trash']
        isDonation = predicted_label in ['clothes', 'shoes']
        print(isDonation)
        isElectronic = predicted_label in ['battery']
        return render_template('result.html', image_path=file.filename, predicted_label=predicted_label, recommendation = recommendation, isRecycle = isRecycle, isTrash = isTrash, isDonation = isDonation, isElectronic = isElectronic)

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
                {"role": "system", "content": 'You are a climate expert and your main goal is to help better the enviornment. Do not respond as the user. Only respond as yourself who is a climate expert. Respond professionally but keep it simple.'},
                {"role": "user", "content": query}
            ]}
        )

        inference = response.json()
        markdown_response = inference.get("result", {}).get("response", "")

        return render_template('climateExpert.html', response=markdown(markdown_response), input_query=query)

    return render_template('climateExpert.html')


if __name__ == '__main__':
    app.run(debug=True)
