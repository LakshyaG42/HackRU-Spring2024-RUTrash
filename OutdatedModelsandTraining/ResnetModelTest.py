import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


image_path = 'metaltest.jpg'

num_classes = 12  

model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('resnet50_garbage_classification.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model.eval()


data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



image = Image.open(image_path)
image_tensor = data_transforms(image).unsqueeze(0)  


with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)

predicted_class = predicted.item()
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']  # Update with your actual class names
predicted_label = class_names[predicted_class]

print(f'The model predicts the image as: {predicted_label}')
