import tensorflow as tf
import pandas as pd
import numpy as np

import os
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

data_dir = "data\garbage_classification"
batch_size = 32
num_classes = len(os.listdir(data_dir))
input_shape = (224, 224, 3) #MobileNetV2 IMAGE INPUT SIZE
gpus = tf.config.list_physical_devices('GPU')
print(f'Using GPU: {gpus}')

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # & of data that we will be using for validation (splits dataset into validation and data)
)
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training' 
)
valid_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation' 
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)


epochs = 10
model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_generator.n // batch_size,
    callbacks=[early_stopping, reduce_lr]
)


model.save('garbage_classification_model.h5')