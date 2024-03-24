import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

num_classes = 12
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']  

model = models.mobilenet_v2()
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


image_path = 'shoestest.jpg'  
image = Image.open(image_path)
image_tensor = data_transforms(image).unsqueeze(0)  #

with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    
predicted_class = predicted.item()
predicted_label = class_names[predicted_class]

print(f'The MobileNetV2 model predicts the image as: {predicted_label}')
