import os
from random import randrange
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Somewhere to store the models
generated_models = []


def label_maker(label):
    if label == '3001':      one_hot = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if label == '3002':      one_hot = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    if label == '3003':      one_hot = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    if label == '3004':      one_hot = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    if label == '3005':      one_hot = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    if label == '3006':      one_hot = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    if label == '3007':      one_hot = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    if label == '3008':      one_hot = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    if label == '3009':      one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    if label == '3010':      one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    return one_hot


# A class to create models and store their data
class Model:

    # These are the most basic parameters passed into the model
    def __init__(self, num_classes, num_layers, epochs, activation, min_filters, max_filters, max_kernel,
                 normalisation):
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.min_filters = min_filters
        self.max_filters = max_filters
        self.normalisation = normalisation
        self.activation = activation
        self.max_kernel = max_kernel
        self.epochs = epochs
        self.model = models.Sequential()

    def __str__(self):
        return self.model.summary()

    # Adding the input layer to the network
    def add_input_layer(self):
        kernel_size = self.choose_kernel()
        num_filters = self.choose_filters()
        self.model.add(layers.Conv2D(num_filters,
                                     (kernel_size, kernel_size),
                                     activation=self.activation,
                                     input_shape=(100, 100, 3)))

    # Add layer to the network
    def add_hidden_layer(self):
        pooling = self.choose_pooling()
        dropout = self.choose_dropout()
        kernel_size = self.choose_kernel()
        num_filters = self.choose_filters()
        # Now we look at adding this layer to the model
        self.model.add(layers.Conv2D(num_filters,
                                     (kernel_size, kernel_size),
                                     activation=self.activation))
        # Depending on the parameters of this layer, these additions may be present or excluded
        if self.normalisation:
            self.model.add(layers.BatchNormalization())
        if dropout > 0:
            self.model.add(layers.Dropout(dropout))
        if pooling > 0:
            self.model.add(layers.MaxPooling2D(pool_size=(pooling, pooling)))  # 12

    # THe dense layers will be added last to allow for the class prediction
    def add_dense_layer(self):
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

    # Compile the model to the structure designed
    def compile_model(self):
        optimizer = "adam"
        loss = "categorical_crossentropy"
        metrics = ['accuracy']
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Fit the model to the dataset given
    def fit_model(self, x, y):
        self.model.fit(x, y, epochs=self.epochs)

    # Randomly assignes the kernel size for a conv layer
    def choose_kernel(self):
        if self.max_kernel > 3:
            return randrange(3, self.max_kernel + 1)
        else:
            return 3

    def get_model(self):
        return self.model
    # Randomly assignes the number of filters for a conv layer
    def choose_filters(self):
        return randrange(self.min_filters, self.max_filters + 1)

    # Randomly assignes the dropout rate for a conv layer up to 0.2
    def choose_dropout(self):
        return randrange(0, 5) / 20

    # Will there be pooling on this layer? yes or no
    def choose_pooling(self):
        return randrange(0, 2)

    # This method creates and saves the model based on randomly chosen variables
    def create_model(self):
        self.add_input_layer()
        for i in range(self.num_layers):
            self.add_hidden_layer()
        self.add_dense_layer()
        self.compile_model()


# Loading in the model

datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=20)
training_dataset = tf.keras.preprocessing.image_dataset_from_directory("datasets/neatly_cropped_pics_2",
                                                                       labels='inferred', label_mode='int',
                                                                       image_size=(100, 100))

dir = "C:/Users/shane t/Desktop/Data/neatly_cropped_pics_2/"
count = 0
x_train = []
y_train = []

for directory in os.listdir(dir):
    for file in os.listdir(dir + directory):
        filename = os.fsdecode(file)
        image = Image.open(dir + directory + '/' + filename)
        # Resize image and convert to array to fit CNN
        resized_image = image.resize((100, 100))
        image_array = tf.keras.preprocessing.image.img_to_array(resized_image).reshape(100, 100, 3)
        x_train.append(image_array)
        y_train.append(label_maker(directory))
        count += 1

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)


# Reminder for the variables passed into models
# num_classes, num_layers, epochs, activation, min_filters, max_filters, max_kernel, normalisation

for i in range(5):

    # Creating the first model:
    current_model = Model(10, 5, 5, 'relu', 5, 40, 5, 1)
    current_model.create_model()
    print(current_model.__str__())
    current_model.fit_model(x_train, y_train)
    model_json = current_model.get_model.to_json()
    with open("models/batch_1/model_" + str(i) + "/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    current_model.get_model.save_weights("models/batch_1/model_" + str(i) + "/model_weights.h5")
    print("Saved model_" + str(i) + "to disk")

print("5 models created!")