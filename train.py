import pathlib
import random

import tensorflow as tf
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE

checkpoint_path = 'training/'
train_path = 'picture/train'
val_path = 'picture/validation'
num_train = 492
num_val = 181
batch_size = 32
epochs = 4
IMG_HEIGHT = 192
IMG_WIDTH = 192


# %%

def load_gen():
    train_image_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.5)

    val_image_gen = ImageDataGenerator(rescale=1./255)

    train_data_gen = train_image_gen.flow_from_directory(
        batch_size=batch_size,
        directory=train_path,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='sparse')

    val_data_gen = val_image_gen.flow_from_directory(
        batch_size=batch_size,
        directory=val_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='sparse')

    return (train_data_gen, val_data_gen)



def create_model():
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False)
    mobile_net.trainable=False

    model = models.Sequential([
        mobile_net,
        layers.GlobalAveragePooling2D(),
        layers.Dense(16, activation = 'softmax')
        # layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        # layers.MaxPooling2D(),
        # layers.Dropout(0.2),
        # layers.Conv2D(32, 3, padding='same', activation='relu'),
        # layers.MaxPooling2D(),
        # layers.Conv2D(64, 3, padding='same', activation='relu'),
        # layers.MaxPooling2D(),
        # layers.Dropout(0.2),
        # layers.Flatten(),
        # layers.Dense(512, activation='relu'),
        # layers.Dense(16, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model



def train(train_data_gen, val_data_gen):
    model = create_model()
    model.summary()

    eachcheckpoint_path = checkpoint_path + "cp-{epoch:01d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=2)

    model_log = model.fit_generator(
        train_data_gen,
        steps_per_epoch=num_train,
        epochs=epochs,
        callbacks=[cp_callback],
        validation_data=val_data_gen,
        validation_steps=num_val
    )
    return model_log


def plot_curve(model_log):
    acc = model_log.history['accuracy']
    val_acc = model_log.history['val_accuracy']

    loss = model_log.history['loss']
    val_loss = model_log.history['val_loss']

    epochs_range = range(2)

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.savefig('pic_A&L.jpg')
    plt.show()


def save_weights(model):
    model.save_weights(checkpoint_path + 'weights')

def load_weights(model):
    model.load_weights(checkpoint_path + 'weights')
    return model

def save_model(model):
    model.save(checkpoint_path + 'model.h5')


# %%

if __name__ == "__main__":
    train_gen, val_gan = load_gen()
    model = create_model()
    # load_weights(model)

    model_log = train(train_gen, val_gan)
    # plot_curve(model_log)

    save_weights(model)
    save_model(model)

    plot_curve(model_log)

