# imports:
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Build model
activaition = 'relu'

# Normal cov layers
model = models.Sequential()
model.add(layers.Conv2D(12, (3, 3), activation=activaition, input_shape=(100, 100, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(16, (3, 3), activation=activaition))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

# 1x1 convolutions
model.add(layers.Conv2D(10, 1, 1, activation=activaition))
# Max pooling (added after first model)
model.add(layers.MaxPooling2D(pool_size=(2, 2))) # 12

# Normal conv layers
model.add(layers.Conv2D(16, (3, 3), activation=activaition))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(20, (3, 3), activation=activaition))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

# 1x1 convolutions
model.add(layers.Conv2D(12, 1, 1, activation=activaition))

# Normal conv layers
model.add(layers.Conv2D(20, (3, 3), activation=activaition))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(24, (3, 3), activation=activaition))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(15, activation = 'softmax'))

optimizer = "adam"
loss = "categorical_crossentropy"
metrics = ['accuracy']
model.compile(optimizer= optimizer,loss = loss, metrics =metrics)

# Load up model and such
model.load_weights("models/Cropped_squished.h5")



labels = ['3001', '3002', '3003', '3004', '3005', '3006', '3007', '3008', '3009', '3010', '3020', '3021', '3022', '3023', '3024']


# Load in some images
directory = "C:/Users/shane t/Desktop/test_dataset/"

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    image = Image.open(directory + filename)

    # Resize image and convert to array to fit CNN
    resized_image = image.resize((100, 100))
    image_array = tf.keras.preprocessing.image.img_to_array(resized_image).reshape(1, 100, 100, 3)

    # Feed image into model
    prediction = model.predict_classes(image_array)
    print("Brick: " + filename + " Prediction: " + labels[int(prediction)])
