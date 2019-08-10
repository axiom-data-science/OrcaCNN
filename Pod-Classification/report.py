#!/usr/bin/env python3
# coding=utf-8

import logging
import argparse
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def classification_report_sklearn(model_path, test_path):
    '''Print Classification report from sklearn'''
    img_width, img_height = 200, 300
    test_data_path = test_path

    test_generator = ImageDataGenerator()
    test_data_generator = test_generator.flow_from_directory(
        test_data_path,
        target_size=(img_width, img_height),
        batch_size=32,
        shuffle=False)
    test_steps_per_epoch = np.math.ceil(
        test_data_generator.samples /
        test_data_generator.batch_size)

    model = load_model(model_path)

    predictions = model.predict_generator(
        test_data_generator, steps=test_steps_per_epoch)

    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = test_data_generator.classes
    class_labels = list(test_data_generator.class_indices.keys())

    report = metrics.classification_report(
        true_classes, predicted_classes, target_names=class_labels)
    print(report)


def main(args):
    model_path = args.modelpath
    test_path = args.testpath

    classification_report_sklearn(model_path, test_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Predict which images belong in which pods")
    parser.add_argument(
        '-m',
        '--modelpath',
        type=str,
        help='path to saved model weights',
        required=True)
    parser.add_argument(
        '-c',
        "--testpath",
        type=str,
        help='directory with Test images',
        required=True)

    args = parser.parse_args()

    main(args)
