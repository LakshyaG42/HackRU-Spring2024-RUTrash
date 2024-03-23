import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

# Load pre-trained ResNet-50 model (excluding the top classification layer)
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers to prevent training
for layer in resnet_model.layers:
    layer.trainable = False

# Define a function to extract features from an image using ResNet-50
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = resnet_model.predict(img_array)
    return features

# Example usage: Compute similarity percentage between two face images
image_path1 = 'white-woman-1.jpg'
image_path2 = 'black-man-1.jpg'

features1 = extract_features(image_path1)
features2 = extract_features(image_path2)

# Compute cosine similarity between the extracted features
similarity_score = np.dot(features1.flatten(), features2.flatten()) / (np.linalg.norm(features1) * np.linalg.norm(features2))

# Map similarity score to percentage scale (0% to 100%)
similarity_percentage = (similarity_score + 1) * 50  # Convert from [-1, 1] range to [0, 100] percentage scale

print("Similarity Percentage:", similarity_percentage)
