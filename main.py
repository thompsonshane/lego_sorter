# imports:
import os

import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# Build model
activaition = 'relu'

model_name = "demo-5_epoch.h5"

# Normal cov layers
model = models.Sequential()
model.add(layers.Conv2D(20, (3, 3), activation=activaition, input_shape=(100, 100, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(25, (3, 3), activation=activaition))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

# 1x1 convolutions
model.add(layers.Conv2D(6, 1, 1, activation=activaition))
# Max pooling
model.add(layers.MaxPooling2D(pool_size=(2, 2))) # 12

# Normal conv layers
model.add(layers.Conv2D(25, (3, 3), activation=activaition))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(30, (3, 3), activation=activaition))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

# 1x1 convolutions
model.add(layers.Conv2D(10, 1, 1, activation=activaition))

# Normal conv layers
model.add(layers.Conv2D(35, (3, 3), activation=activaition))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(40, (3, 3), activation=activaition))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

optimizer = "adam"
loss = "categorical_crossentropy"
metrics = ['accuracy']
model.compile(optimizer= optimizer,loss = loss, metrics =metrics)

model.summary()


# Load up model and such
model.load_weights("models/demo-5_epoch.h5")

labels = ['3001', '3002', '3003', '3004', '3005', '3006', '3007', '3008', '3009', '3010']

# Load in some images
directory = "datasets/from_pi/"

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    image = Image.open(directory + filename)

    # Resize image and convert to array to fit CNN
    resized_image = image.resize((100, 100))
    image_array = tf.keras.preprocessing.image.img_to_array(resized_image).reshape(1, 100, 100, 3)

    # Feed image into model
    prediction = model.predict_classes(image_array)
    print("Brick: " + filename + " Prediction: " + labels[int(prediction)])
