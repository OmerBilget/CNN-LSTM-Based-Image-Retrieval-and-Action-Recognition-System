import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 3)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.reshape(x_train.shape[0], 32,32,3)
x_test = x_test.reshape(x_test.shape[0], 32,32,3)
datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
    #zoom_range=0.3
    )


## model experiments

class AutoencoderV1:
    def __init__(self):
        self.input_shape = input_shape
    def build_model_full(self):
        input_image = keras.Input(shape=input_shape)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_image)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoder_output = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoder_output)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        decoder_output = layers.Conv2D(3, (3, 3), padding='same')(x)
        model_encoder = keras.Model(input_image, encoder_output)
        model_full = keras.Model(input_image, decoder_output)
        return model_encoder, model_full

class AutoencoderV2:
    def __init__(self,latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
    def build_model_full(self):
        input_image = keras.Input(shape=input_shape)
        x = layers.Conv2D(12, (3, 3), activation='relu', padding='same',strides=2)(input_image)
        x = layers.Conv2D(24, (3, 3), activation='relu', padding='same',strides=2)(x)
        x = layers.Conv2D(48, (3, 3), activation='relu', padding='same', strides=2,name='encoder')(x)

        x = layers.Convolution2DTranspose(48, (3, 3), activation='relu', padding='same',strides=2)(x)
        x = layers.Convolution2DTranspose(24, (3, 3), activation='relu', padding='same',strides=2)(x)
        x = layers.Convolution2DTranspose(12, (3, 3), activation='relu', padding='same',strides=2)(x)
        decoder_output = layers.Conv2D(3, (3, 3), activation='softmax', padding='same', strides=1,name="decoded")(x)

        model = keras.Model(input_image, decoder_output)
        return model

class AutoencoderV3:
    def __init__(self,latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
    def build_model_full(self):
        #encoder
        input_image = keras.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',strides=(2,2))(input_image)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',strides=(2,2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2,2),name='encoder')(x)
        x = layers.BatchNormalization()(x)

        # img_shape = x.shape[1:]
        # x = layers.Flatten()(x)
        # x = layers.Reshape(img_shape)(x)

        #decoder
        x = layers.Convolution2DTranspose(32, (3, 3), activation='relu', padding='same',strides=(2,2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Convolution2DTranspose(64, (3, 3), activation='relu', padding='same',strides=(2,2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Convolution2DTranspose(32, (3, 3), activation='relu', padding='same',strides=(2,2))(x)
        x = layers.BatchNormalization()(x)
        decoder_output = layers.Conv2D(3, (3, 3), activation='softmax', padding='same', strides=1,name="decoded")(x)
        model = keras.Model(input_image, decoder_output,name="autoencoder")
        return model
a2=AutoencoderV2(256)
model= a2.build_model_full()

"""
## Train the model
"""
model.summary()

batch_size = 128
epochs =30


model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, x_test))


score = model.evaluate(x_test, x_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

decoded_imgs = model.predict(x_test)


model.save("autoencoder.keras")
