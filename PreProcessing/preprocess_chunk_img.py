#!/usr/bin/env python3
# coding=utf-8

# Imports

import os
import argparse
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

from skimage.restoration import (denoise_wavelet, estimate_sigma)
from pydub import AudioSegment
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Supress matplotlib warnings
plt.rcParams.update({'figure.max_open_warning': 0})


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


def apply_per_channel_energy_norm(data, sampling_rate):
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


def set_rate(audio, rate):
    return audio.set_frame_rate(rate)


def make_chunks(filename, chunk_size, sampling_rate, target_location):
    '''Divide the audio file into chunk_size samples'''
    f = AudioSegment.from_wav(filename)

    if f.frame_rate != sampling_rate:
        f = set_rate(f, sampling_rate)

    j = 0

    if not os.path.exists(target_location):
        os.makedirs(target_location)

    os.chdir(target_location)

    # Get file name
    f_name = os.path.basename(filename)

    while len(f[:]) >= chunk_size * 1000:
        chunk = f[:chunk_size * 1000]
        chunk.export(f_name[:-4] + "_{:04d}.wav".format(j), format="wav")
        logger.info("Padded file stored as " + f_name[:-4] + "_{:04d}.wav".format(j))
        # print("File stored at " + f_name[:-4] + "_{:04d}.wav".format(j))
        f = f[chunk_size * 1000:]
        j += 1

    if 0 < len(f[:]) and len(f[:]) < chunk_size * 1000:
        silent = AudioSegment.silent(duration=chunk_size * 1000)
        paddedData = silent.overlay(f, position=0, times=1)
        paddedData.export(f_name[:-4] + "_{:04d}.wav".format(j), format="wav")
        logger.info("Padded file stored as " + f_name[:-4] + "_{:04d}.wav".format(j))
        # print("File stored at " + f_name[:-4] + "_{:04d}.wav".format(j))


def plot_and_save(denoised_data, f_name):

    fig, ax = plt.subplots()

    i = 0
    # Add this line to show plots else ignore warnings
    # plt.ion()

    ax.imshow(denoised_data)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.set_size_inches(10, 10)
    fig.savefig(
        f"{f_name[:-4]}" + "_{:04d}.png".format(i),
        dpi=80,
        bbox_inches="tight",
        quality=95,
        pad_inches=0.0)

    fig.canvas.draw()
    fig.canvas.flush_events()
    i += 1


def standardize_and_plot(sampling_rate, file_path_image):
    logger.info(f"All files will be resampled to {sampling_rate}Hz")

    output_image_folder = "PreProcessed_image/"

    for dirs, subdirs, files in os.walk(file_path_image):
        for i, file in enumerate(files):
            if file.endswith(('.wav', '.WAV')):
                logger.info(f"Pre-Processing file: {file}")
                data, sr = librosa.core.load(
                    os.path.join(dirs, file), sr=sampling_rate, res_type='kaiser_fast')
                target_path = os.path.join(output_image_folder, dirs)

                # There is no need to apply padding since all samples are of same length
                # apply padding
                # padded_data = padding(data, input_length)

                # TODO: mismatch of shape
                # if use_mfcc:
                #     mfcc_data = mfcc(padded_data, sampling_rate, n_mfcc)
                # else:
                #     mfcc_data = preprocessing_fn(padded_data)[:, np.newaxis]

                pcen_S = apply_per_channel_energy_norm(data, sr)

                denoised_data = wavelet_denoising(pcen_S)

                # Get Current working directory to make parent folders
                work_dir = os.getcwd()

                if not os.path.exists(target_path):
                    os.makedirs(target_path)

                os.chdir(target_path)

                # Get file name
                f_name = os.path.basename(file)

                # Plotting and Saving
                plot_and_save(denoised_data, f_name)

                # Change to parent directory to make parent sub-folders
                os.chdir(work_dir)


def main(args):
    sampling_rate = args.resampling
    audio_duration = args.dur
    use_mfcc = args.mfcc
    n_mfcc = args.nmfcc
    file_path_audio = args.classpath
    chunkSize = args.chunks

    audio_length = sampling_rate * audio_duration
    def preprocessing_fn(x): return x
    input_length = audio_length

    no_of_files = len(os.listdir('.'))

    output_audio_folder = "PreProcessed_audio/"

    # Traverse all files inside each sub-folder and make chunks of audio file
    for dirs, subdirs, files in os.walk(file_path_audio):
        for file in files:
            if file.endswith(('.wav', '.WAV')):
                logger.info(f"Making chunks of size {chunkSize}s of file: {file}")

                input_file = os.path.join(dirs, file)

                work_dir = os.getcwd()

                output_path = os.path.join(output_audio_folder, dirs)

                '''
                CouldntDecodeError: Decoding failed. ffmpeg returned error
                code: 1 in file ._20180605_0645_AD8.wav 2018, so catching exception
                '''
                try:
                    make_chunks(
                        input_file,
                        chunkSize,
                        sampling_rate,
                        output_path)
                except Exception as e:
                    logger.error(f"Exception: {e}", exc_info=True)
                    pass

                # Change to parent directory to make parent sub-folders
                os.chdir(work_dir)

    # file_path now directs to the path with all the audio chunks
    file_path_image = os.path.join(output_audio_folder, file_path_audio)

    logger.info(f"Starting to load {no_of_files} data files in the directory")

    standardize_and_plot(sampling_rate, file_path_image)


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
        '-r',
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
        '-s',
        "--chunks",
        type=int,
        default=5,
        help='Chunk Size for each sample to be divided to')
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
