import numpy as np
import pandas as pd
import tensorflow as tf
#import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
#from keras.utils.vis_utils import plot_model
from tensorflow.keras.layers import Flatten, Dense, AvgPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

# Importing the ImageDataGenerator for pre-processing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#print(tf.test.is_gpu_available())
#print("***********************************************")
#print(tf.config.list_physical_devices('GPU'))
EPOCHS = 12
BATCH_SIZE = 128
INPUT_SHAPE = [224, 224, 3]

TRAIN_DIR = "/scratch/scratch6/gopi/gopi/TrainValSplitDataset/train"
VAL_DIR = "/scratch/scratch6/gopi/gopi/TrainValSplitDataset/val"

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

    #x = Flatten()(vgg16.output)
    x = AvgPool2D(pool_size=(4,4),strides = (4,4))(vgg16.output)
    x = Flatten()(x)

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
        rescale=1.0 / 255,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15, preprocessing_function=preprocess_input
    )
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255, preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),  # target_size = input image size
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(224, 224),  # target_size = input image size
        batch_size=BATCH_SIZE,
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

    es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
    mc = ModelCheckpoint('/scratch/scratch6/gopi/gopi/Models/VGG16_Model/VGG16_VOC2012_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)

    history = model.fit_generator(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks = [mc,es],
        verbose = 1

    )

    return model, history


if __name__ == "__main__":
    print("Started..!!")

    new_model = create_model()
    train_generator, val_generator = pre_process_data()
    result_model, history = train_model(new_model, train_generator, val_generator)

    # result_model.save(
    #     "/scratch/scratch6/gopi/gopi/Models/VGG16_Model/VGG16_VOC2012_model.h5"
    # )
    print("Successfully saved the model.")

    # new_model = load_model('/scratch/scratch2/gopi/Models/VGG Model/VGG16_VOC2012_model.h5')
    # new_model.summary()
    # plot_model(new_model)
