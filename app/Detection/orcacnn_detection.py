#!/usr/bin/env python3
# coding=utf-8

import os

import argparse
import logging
import logging.config
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# Disable PIL.PngImagePlugin DEBUG logs
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, img_width, img_height, train_data_path, validation_data_path):
    '''Train the Detection model'''
    nb_train_samples = sum(len(files)
                           for _, _, files in os.walk(train_data_path))
    nb_validation_samples = sum(len(files)
                                for _, _, files in os.walk(validation_data_path))

    epochs = 250
    batch_size = 256
    checkpoint = ModelCheckpoint(filepath='checkpoint_orcacnn-{epoch:02d}-{val_loss:.2f}.h5',
                                 monitor='val_loss', verbose=0, save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=100, min_lr=1e-8)

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2)

    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Change the batchsize according to your system RAM
    train_batchsize = 256
    val_batchsize = 256

    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=(img_width, img_height),
        batch_size=train_batchsize,
        class_mode='binary',
        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_path,
        target_size=(img_width, img_height),
        batch_size=val_batchsize,
        class_mode='binary',
        shuffle=False)

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint, reduce_lr])

    model.save('orca_detection_adam_.h5')

    logger.info("Detection Model saved")


class OrcaNet:
    @staticmethod
    def build(img_width, img_height):
        '''Build CNN for Orca Detection'''
        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)

        model = Sequential()
        model.add(Conv2D(32, (5, 5), padding='same', strides=2, input_shape=input_shape))
        model.add(Conv2D(32, (5, 5), padding='same', strides=2))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Conv2D(32, (5, 5), strides=2, padding='same'))
        model.add(Conv2D(32, (5, 5), strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Conv2D(64, (5, 5), strides=2, padding='same'))
        model.add(Conv2D(64, (5, 5), strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, (5, 5), strides=2, padding='same'))
        model.add(Conv2D(64, (5, 5), strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.summary()

        return model


def main(args):
    dataset_path = args.classpath

    train_data_path = os.path.join(dataset_path, 'train_orca/')
    validation_data_path = os.path.join(dataset_path, 'val_orca/')

    img_width, img_height = 200, 300

    logger.info("Starting compiling of OrcaNet ... ")
    model = OrcaNet.build(img_width=img_width, img_height=img_height)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=3e-5),
                  metrics=['accuracy'])
    logger.info("Starting Training ... ")
    train(model=model, img_width=img_width, img_height=img_height, train_data_path=train_data_path,
          validation_data_path=validation_data_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train CNN model for detection of orca calls in spectrograms")
    parser.add_argument(
        '-c',
        '--classpath',
        type=str,
        help='directory with pos and neg samples in two respective folders',
        required=True)

    args = parser.parse_args()

    main(args)
