# sign_language_recognition.py
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


data_path = r"C:\Users\uasr\Downloads\archive\asl_dataset" 
labels = os.listdir(data_path)

images = []
image_labels = []


for label in labels:
    folder = os.path.join(data_path, label)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))
        images.append(img)
        image_labels.append(label)

images = np.array(images)
image_labels = np.array(image_labels)


le = LabelEncoder()
y = le.fit_transform(image_labels)


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(images, y, epochs=150, batch_size=32, validation_split=0.2)


model.save("asl_model.h5")
print("Model saved as 'asl_model.h5'")


def predict_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found")
        return
    img = cv2.resize(img, (64,64))
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred_label = le.inverse_transform([np.argmax(pred)])
    print("Predicted:", pred_label[0])


predict_image(r"C:\Users\uasr\Downloads\archive\asl_dataset\A\img1.jpg")  