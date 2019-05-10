# OrcaCNN: Detecting and Classifying Killer Whale from Acoustic Data

Passive acoustic observation of whales is an increasingly important tool for whale research. Accurately detecting whale sounds and correctly classifying them into corresponding whale pods are essential tasks, especially in the case when two or more species of whales vocalize in the same observed area. Most of the current tasks of whale sound detection and classification still need to be implemented manually.

This project aims to develop two deep learning models for the detection and pod-classification of orca, or killer whale calls in unknown long audio samples. These deep neural networks will help identify and verify killer whale calls so that researchers, grad students, and shipping vessels don't have to. The end-user interface is made as a web-app which can easily be used by scientists in their research.

## Pipeline

<p align = "center">
<img src = assets/pipeline.jpg>
</p>

## Implementation

There are mainly three stages involved in the development of OrcaCNN.

- Data preparation and Pre-processing
- Hyperparameter Tuning and Training with Model Evaluation
- Deployment

Sufficient steps have been mentioned for each of these stages to aid in the development of OrcaCNN model. Each method has its merits and demerits. It is therefore essential to evaluate your choice before proceeding with any method.

The methods chosen for this project have been mentioned besides each method.

### Data preparation and Pre-processing

The real-world data is often noisy and requires pre-processing. To tackle this, following are some pre-processing steps to put all the samples in a standard format.

- Resampling and normalization of all the audio samples: Since the
sampling rates of all the audio samples will be different, this step will
resample all the audio files to a specific sampling frequency.
- [Data Augmentation](https://medium.com/@alibugra/audio-data-augmentation-f26d716eee66) techniques for padding of samples.
- Removing all the dead samples (negligible frequency)(if any) from the
dataset.
- Denoising: The [Spectral-Subtraction](https://doi.org/10.1109/TASSP.1979.1163209) method can be used to reduce background
noise. This method is based on spectral averaging and residual
noise reduction, widely used for enhancement of noisy speech
signals and can remove the stationary noise included in the sound.
[Sample Code](https://github.com/tracek/Ornithokrites/blob/master/noise_subtraction.py). The Wiener filter is also an option.
- The spectrogram images for each audio file is prepared with Linear Short Time Fourier
Transform/Log-Mel Filter Bank features using Python. [Librosa](https://librosa.github.io/librosa/index.html) will be the
preferred choice for computing these feature representations here.
- Contrast Enhancement: [From past experience](https://github.com/jaimeps/whale-sound-classification/blob/master/7_additional_feature_design/Alternative_image_preprocessing.ipynb), Histogram Equalization
has been found as a better option than capping the extreme values to
mean ± 1.5 std.
- [Focusing on Low-Frequency Spectrum](https://github.com/jaimeps/whale-sound-classification/blob/master/4_image_preprocessing_and_template_extraction_tutorial/Tutorial_image_preprocessing.ipynb): The whale calls are normally
found in the lower frequency spectrum ranging from 100Hz-200Hz.
This would allow us to look only at the specific part of the image which
will be beneficial to the CNN architecture when given as input. The
rest of the image part would roughly count as “noise” (or irrelevant)
portion for CNN.
- Signal to Noise (SNR) Ratio: It is essential to have a high value of SNR for
all the audio samples. [Previous kaggle competition](http://blog.kaggle.com/2013/05/06/summary-of-the-whale-detection-competition/) have shown this as an essential factor in improving the model accuracy.
- [Hydrophone data are subject to a variety of in-band noise sources](https://asa.scitation.org/doi/am-pdf/10.1121/1.5054911?class=chorus+notVisible). A
band-pass filter is a simple way to remove unwanted noise outside of
the signal frequency band. Wavelet denoising is an effective method
for SNR improvement in environments with wide range of noise types
competing for the same subspace.
- To improve robustness to loudness variation, per-channel energy
normalization (PCEN) was [found better than](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/8a75d472dc7286653a5245a80a7603a1db308af0.pdf) the standard log-mel
frontend (and even with an optimized spectral subtraction process).
This provided a 24% reduction in error rate of whale call detection. It
also helps in reducing the narrow-band noise which is most often
caused by nearby boats and the equipment itself.
- [The following paper](https://arxiv.org/pdf/1706.07156.pdf) shows that Mel-scaled STFT outperforms other methods like
Constant Q-Transform(CQT) and Constant Wavelet Transform (CWT).
**Architecturally speaking, 2D convolutions were found to perform better than
1D convolutions.**