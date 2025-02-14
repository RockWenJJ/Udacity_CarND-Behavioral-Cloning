import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Activation, Flatten,Dropout, Input, Lambda, GlobalAveragePooling2D, Cropping2D, Conv2D
from sklearn.model_selection import train_test_split
import os
import csv
import cv2
import sklearn
import random

def get_data(path):
    samples = []
    with open(os.path.join(path, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        idx = 0
        for line in reader:
            if idx>0:
                samples.append(line)
            idx += 1
    train_samples, valid_samples = train_test_split(samples, test_size = 0.2)
    return train_samples, valid_samples

def generator(samples, path, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            offset_end = offset+batch_size if offset+batch_size < num_samples else num_samples
            batch_samples = samples[offset:offset_end]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = os.path.join(path, 'IMG/'+batch_sample[0].split('/')[-1])
                l_name = os.path.join(path, 'IMG/'+batch_sample[1].split('/')[-1])
                r_name = os.path.join(path, 'IMG/'+batch_sample[2].split('/')[-1])
                try:
                    center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                    left_image = cv2.cvtColor(cv2.imread(l_name), cv2.COLOR_BGR2RGB)
                    right_image = cv2.cvtColor(cv2.imread(r_name), cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(name)
                    exit(e)
                correction = 0.2
                center_angle = float(batch_sample[3])
                right_angle = float(batch_sample[3] - correction)
                left_angle = float(batch_sample[3] + correction)
                images.extend([center_image, np.fliplr(center_image), right_image, np.fliplr(right_image), left_image, np.fliplr(left_image)])
                angles.extend([center_angle, -center_angle, right_angle, -right_angle, left_angle, -left_angle])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train

def pilotnet():
    image = Input(shape=(160, 320, 3))
    net = Cropping2D(cropping=((50, 20), (0, 0)))(image)
    net = Lambda(lambda image:tf.image.resize_images(image, (66, 200)))(net)
    net = Lambda(lambda image:image/127.5 - 1.)(net)

    net = Conv2D(filters = 24, kernel_size = (5, 5), strides = 2, activation = "relu")(net)
    net = Conv2D(filters = 36, kernel_size = (5, 5), strides = 2, activation = "relu")(net)
    net = Conv2D(filters = 48, kernel_size = (5, 5), strides = 2, activation = "relu")(net)
    net = Conv2D(filters = 64, kernel_size = (3, 3), strides = 1, activation = "relu")(net)
    net = Conv2D(filters = 64, kernel_size = (3, 3), strides = 1, activation = "relu")(net)

    net = Flatten()(net)
    net = Dense(1164, activation = "relu")(net)
    net = Dropout(0.5)(net)
    net = Dense(100, activation = "relu")(net)
    net = Dropout(0.5)(net)
    net = Dense(10, activation = "relu")(net)
    net = Dropout(0.5)(net)
    pred = Dense(1, activation = "tanh")(net)

    model = Model(inputs=image, outputs=pred)
    model.summary()
    return model

def resnet50():
    resnet = ResNet50(weights="imagenet", include_top=False,
                      input_shape=(224, 224, 3))
    for layer in resnet.layers:
        layer.trainable = False

    # input placeholder
    image = Input(shape=(160, 320, 3))
    img_crop = Cropping2D(cropping=((50, 20), (0, 0)))(image)
    img_rsz = Lambda(lambda image: tf.image.resize_images(
        image, (224, 224)))(img_crop)
    img_nrm = Lambda(lambda image: image / 127.5 - 1.)(img_rsz)

    net = resnet(img_nrm)
    net = GlobalAveragePooling2D()(net)
    net = Dense(512, activation="relu")(net)
    net = Dropout(0.5)(net)
    pred = Dense(1, activation="tanh")(net)

    model = Model(input=image, output=pred)
    model.summary()

    return model

if __name__ == "__main__":
    path = "/data/DeepLearning/github_projects/Udacity_Project/beta_simulator_linux/data_0"
    batch_size = 32
    save_path = "models"
    train_samples, valid_samples = get_data(path)
    train_generator = generator(train_samples, path, batch_size)
    valid_generator = generator(valid_samples, path, batch_size)

    model = pilotnet()
    # model = resnet50()

    model.compile(loss='mse', optimizer='adam', metrics = ["accuracy"])
    checkpoint = ModelCheckpoint(filepath=save_path, monitor="val_loss", save_best_only=True)
    stopper = EarlyStopping(monitor="val_loss", min_delta=2e-5, patience=10)
    history_object = model.fit_generator(train_generator, steps_per_epoch = np.ceil(len(train_samples) / batch_size), validation_data = valid_generator,
                        validation_steps = np.ceil(len(valid_samples) / batch_size),epochs = 50, verbose = 1, callbacks=[checkpoint, stopper])

    model.save("model.h5")

    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()