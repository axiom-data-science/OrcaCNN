#!/usr/bin/env python3
# coding=utf-8

import logging
import argparse
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn.metrics import confusion_matrix

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
    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0.5] = 1

    predicted_classes = (predictions).astype(np.int)

    true_classes = test_data_generator.classes
    class_labels = list(test_data_generator.class_indices.keys())

    report = metrics.classification_report(
        true_classes, predicted_classes, target_names=class_labels)
    print(report)

    cm = confusion_matrix(true_classes, predicted_classes)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("Accuracy: {:.4f}".format(acc))
    print("Sensitivity: {:.4f}".format(sensitivity))
    print("Specificity: {:.4f}".format(specificity))


def main(args):
    model_path = args.modelpath
    test_path = args.testpath

    classification_report_sklearn(model_path, test_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Predict which images are orcas")
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
