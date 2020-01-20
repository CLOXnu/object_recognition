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
IMG_HEIGHT = 192
IMG_WIDTH = 192


# %%

def load_model():
    model = models.load_model('dataset_training/model.h5')
    return model

def load_labelname(path):
    root = pathlib.Path(path)
    labels = sorted(root_item.name for root_item in root.glob('*/') if root_item.is_dir())
    return labels

def load_image(path, resizeTo):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, resizeTo)
    image /= 255.0
    return image

def show_image(image):
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    plt.grid(False)
    plt.show()


def predict(model, path, labels, img_height, img_width):
    test_img = load_image(path, resizeTo=[img_height, img_width])
    test_img = np.reshape(test_img, (1, img_height, img_width, 3))
    pre = model.predict(test_img)
    print(pre)
    print("predict: ", np.argmax(pre))
    print("predict: ", labels[np.argmax(pre)])
    return pre


def plot_pre(pre, labels):
    plt.grid(False)
    plt.xticks(range(len(pre)))
    plt.yticks([])
    thisplot = plt.bar(range(len(pre)), pre, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(pre)
    thisplot[predicted_label].set_color('blue')
    for i in range(len(pre)):
        plt.text(i, pre[i], '%.1f %%' % (pre[i]*100), ha='center', va='bottom')

    plt.figure(1)
    plt.show()


# %%

def create_model():
    # mobile_net = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False)
    # mobile_net.trainable=False

    model = models.Sequential([
        # mobile_net,
        # layers.GlobalAveragePooling2D(),
        # layers.Dense(16, activation = 'softmax')
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(16, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


def save_weights(model):
    model.save_weights(checkpoint_path + 'weights')

def load_weights(model):
    model.load_weights(checkpoint_path + 'weights')
    return model



# %%

if __name__ == "__main__":
    resizeTo = [IMG_HEIGHT, IMG_WIDTH]
    class_path = 'picture/all'
    img_path = 'picture/test/0.jpg'

    labels = load_labelname(class_path)
    img = load_image(img_path, resizeTo)
    show_image(img)

    # model = create_model()
    # model = load_weights(model)
    model = load_model()
    pre = predict(model, img_path, labels, IMG_HEIGHT, IMG_WIDTH)
    print(list((index, label) for (index, label) in enumerate(labels)))
    plot_pre(pre[0], labels)


