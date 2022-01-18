#!/usr/bin/env python3
# coding=utf-8

import shutil
import os
import logging
import argparse
import csv
from itertools import zip_longest
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_predictions(model_path, folder_path):
    '''
    Stacking images eats up RAM much faster: https://hjweide.github.io/efficient-image-loading
    '''
    img_width, img_height = 200, 300
    N = sum(len(files) for _, _, files in os.walk(folder_path))
    data = np.empty((N, img_width, img_height, 3), dtype=np.uint8)

    for dirs, _, files in os.walk(folder_path):
        for i, file in enumerate(files):
            f_name = os.path.join(dirs, file)
            img = image.load_img(f_name, target_size=(img_width, img_height))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            # images.append(img)
            data[i, ...] = img

    if os.path.isfile(model_path):
        model = load_model(model_path)
        classes = (model.predict(data) > 0.5).astype("int32")
    elif os.path.isdir(model_path):
        scores = np.zeros((N, 1))
        for model_name in os.listdir(model_path):
            model = load_model(os.path.join(model_path, model_name))
            scores += (model.predict(data) > 0.5).astype("int32")
        classes = scores / len(os.listdir(model_path))
        classes[classes < 0.5] = 0
        classes[classes >= 0.5] = 1
        classes = classes.astype("int32")
    else:
        raise ValueError(f"Path provided, {model_path}, is not a valid path!")

    return classes


def predict(model_path, test_path):
    folder_path = test_path
    model_path = model_path

    logger.info("Starting Prediction")

    classes = generate_predictions(model_path, folder_path)

    # stack up images list to pass for prediction
    # images = []
    # images = np.vstack(images)
    # classes = model.predict_classes(data)

    f, pos, negs = [], [], []
    for i in os.listdir(folder_path):
        f.append(i)

    for i in range(len(classes)):
        f_n = os.path.join(folder_path, f[i])

        os.makedirs("pos_orca", exist_ok=True)
        if classes[i][0] == 1:
            pos.append(f[i])
            shutil.copy(f_n, 'pos_orca')
        else:
            negs.append(f[i])

    with open(os.path.join("pos_orca", "predictions.csv"), "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["high_confidence", "low_confidence"])
        for values in zip_longest(*[pos, negs]):
            writer.writerow(values)

    logger.info(
        f"Detected {sum(len(files) for _, _, files in os.walk('pos_orca'))} orca calls")


def rename_for_template_matching(test_path):
    '''Renames files to aid in template matching process'''
    file_l = os.listdir(test_path)
    file_list = file_l[1:]

    for index, file in enumerate(file_list):
        os.rename(
            os.path.join(
                test_path, file), os.path.join(
                    test_path, str(index) + '.png'))


def main(**kwargs):
    model_path = kwargs['modelpath']
    test_path = kwargs['testpath']
    rename_for_template_matching(test_path)
    predict(model_path, test_path)


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
        help='directory with PreProcessed images',
        required=True)

    kwargs = parser.parse_args()

    main(**kwargs)
