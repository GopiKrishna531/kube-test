import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.utils.vis_utils import plot_model
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.models import load_model

# Importing the ImageDataGenerator for pre-processing
from keras.preprocessing.image import ImageDataGenerator


EPOCHS = 7
BATCH_SIZE = 32
INPUT_SHAPE = [224, 224, 3]

TRAIN_DIR = "/scratch/scratch2/gopi/TrainValSplitDataset/train"
VAL_DIR = "/scratch/scratch2/gopi/TrainValSplitDataset/val"

IMAGE_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
TOTAL_IMAGE_CLASSES = len(IMAGE_CLASSES)


def create_model():

    vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    # vgg16.summary()
    # plot_model(vgg16)

    # this will exclude the initial layers from training phase as there are already been trained.
    for layer in vgg16.layers:
        layer.trainable = False

    x = Flatten()(vgg16.output)

    # we can add a new fully connected layer but it will increase the execution time.
    x = Dense(128, activation="relu")(x)

    # adding the output layer with softmax function as this is a multi label classification problem
    x = Dense(
        TOTAL_IMAGE_CLASSES,
        activation="softmax",
    )(x)

    # x = Dense(1024,activation='relu',name='My_fc')(vgg16.layers[-2].output)
    # x = Dense(TOTAL_IMAGE_CLASSES,activation='softmax',name = 'predictions')(x)

    model = Model(inputs=vgg16.input, outputs=x)
    # model.summary()
    # plot_model(model)

    return model


def pre_process_data():

    # Initialising the generators for train and test data
    # The rescale parameter ensures the input range in [0, 1]

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, preprocessing_function=preprocess_input
    )
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255, preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),  # target_size = input image size
        batch_size=32,
        class_mode="categorical",
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(224, 224),  # target_size = input image size
        batch_size=32,
        class_mode="categorical",
    )
    print("**** Train generator calss indices *****")
    train_generator.class_indices
    print("**** Val generator calss indices *****")
    val_generator.class_indices

    return train_generator, val_generator


def train_model(model, train_generator, val_generator):
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=val_generator,
    )

    return model, history


if __name__ == "__main__":
    print("Started..!!")

    new_model = create_model()
    train_generator, val_generator = pre_process_data()
    result_model, history = train_model(new_model, train_generator, val_generator)

    result_model.save(
        "/scratch/scratch2/gopi/Models/VGG16_Model/VGG16_VOC2012_model.h5"
    )
    print("Successfully saved the model.")

    # new_model = load_model('/scratch/scratch2/gopi/Models/VGG Model/VGG16_VOC2012_model.h5')
    # new_model.summary()
    # plot_model(new_model)
