#!/usr/bin/python
# coding=utf-8

# Imports

import os
import glob
import argparse
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

from skimage.restoration import (denoise_wavelet, estimate_sigma)


def padding(data, input_length):
    '''Padding of samples to make them of same length'''
    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length + offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
    return data


def audio_norm(data):
    '''Normalization of audio'''
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 1e-6)
    return data - 0.5


def mfcc(data, sampling_rate, n_mfcc):
    '''Compute mel-scaled feature using librosa'''
    data = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
    # data = np.expand_dims(data, axis=-1)
    return data


def pcen(data, sampling_rate):
    '''Compute Per-Channel Energy Normalization (PCEN)'''
    S = librosa.feature.melspectrogram(
        data, sr=sampling_rate, power=1)  # Compute mel-scaled spectrogram
    # Convert an amplitude spectrogram to dB-scaled spectrogram
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    pcen_S = librosa.core.pcen(S)
    return pcen_S


def wavelet_denoising(data):
    '''
    Wavelet Denoising using scikit-image
    NOTE: Wavelet denoising is an effective method for SNR improvement in environments with
              wide range of noise types competing for the same subspace.
    '''
    sigma_est = estimate_sigma(data, multichannel=True, average_sigmas=True)
    im_bayes = denoise_wavelet(data, multichannel=False, convert2ycbcr=True, method='BayesShrink',
                               mode='soft')
    im_visushrink = denoise_wavelet(data, multichannel=False, convert2ycbcr=True, method='VisuShrink',
                                    mode='soft')

    # VisuShrink is designed to eliminate noise with high probability, but this
    # results in a visually over-smooth appearance. Here, we specify a reduction
    # in the threshold by factors of 2 and 4.
    im_visushrink2 = denoise_wavelet(data, multichannel=False, convert2ycbcr=True, method='VisuShrink',
                                     mode='soft', sigma=sigma_est / 2)
    im_visushrink4 = denoise_wavelet(data, multichannel=False, convert2ycbcr=True, method='VisuShrink',
                                     mode='soft', sigma=sigma_est / 4)
    return im_bayes


def main(args):
    sampling_rate = args.resampling
    audio_duration = args.dur
    use_mfcc = args.mfcc
    n_mfcc = args.nmfcc
    file_path = args.classpath

    audio_length = sampling_rate * audio_duration
    def preprocessing_fn(x): return x
    input_length = audio_length

    # Output folder where pre-processed files will be saved
    output_path = "PreProcessed"

    # Load files
    os.chdir(file_path)
    no_of_files = len(os.listdir('.'))

    print(f"Starting to load {no_of_files} data files in the directory")
    print(f"All files will be resampled to {sampling_rate}Hz with audio duration {audio_duration}s")

    for i, file in enumerate(glob.glob("*.wav")):
        print(f"Pre-Processing file: {file}")
        data, sr = librosa.core.load(file, sr=sampling_rate, res_type='kaiser_fast')

        # apply padding
        padded_data = padding(data, input_length)

        # TODO: mismatch of shape
        # if use_mfcc:
        #     mfcc_data = mfcc(padded_data, sampling_rate, n_mfcc)
        # else:
        #     mfcc_data = preprocessing_fn(padded_data)[:, np.newaxis]

        # apply Per-Channel Energy Normalization
        pcen_S = pcen(padded_data, sr)

        # apply Wavelet Denoising
        denoised_data = wavelet_denoising(pcen_S)

        # Plotting and Saving
        fig, ax = plt.subplots()
        plt.ion()

        # make directory if it does not exist
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        ax.imshow(denoised_data)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.set_size_inches(10, 10)
        fig.savefig(
            f"{output_path}/spec_denoised{i}.png",
            dpi=80,
            bbox_inches="tight",
            quality=95,
            pad_inches=0.0)
        fig.canvas.draw()
        fig.canvas.flush_events()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Pre-Process the audio files and save as spectrogram images")
    parser.add_argument(
        '-c',
        '--classpath',
        type=str,
        help='directory with list of classes',
        required=True)
    parser.add_argument(
        '-s',
        '--resampling',
        type=int,
        default=44100,
        help='choose sampling rate')
    parser.add_argument(
        '-d',
        "--dur",
        type=int,
        default=2,
        help='Max duration (in seconds) of each clip')
    parser.add_argument(
        '-m',
        "--mfcc",
        type=bool,
        default=False,
        help='apply mfcc')
    parser.add_argument(
        '-n',
        "--nmfcc",
        type=int,
        default=20,
        help='Number of mfcc to return')

    args = parser.parse_args()

    main(args)
