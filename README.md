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

### Data Description

The data was given by [Dan Olsen](https://www.zegrahm.com/field-leaders/dan-olsen) from his research work. The data being around 1 TB was uploaded to a Google Cloud Bucket where the files could directly be accessed. There were several [approaches taken for transferring data to Google Colab](https://github.com/axiom-data-science/OrcaCNN/wiki/Transferring-data#approaches-taken-for-transferring-data-to-google-and-making-it-available-in-colab) and thanks to [Shane](https://github.com/shane-axiom) for writing them down graciously.

The data consisted of two folders:

- Autonomous Recordings NGOS
  - E3
  - E4
  - H2
  - L1
  - L2
  - P2

Most of the autonomous recordings are very quiet, which improves the quality and the distance with which the whale calls can be detected (less distortion). The autonomous recordings need to be amplified, or sensitized, for the spectrograms to capture the whale calls properly. But these are not labelled, and hence cannot be used further except for testing data.

As mentioned by Dan, most of the E3, E4, H2, L1 directories do not have whale calls to aid in automating the process of finding the calls out of a lot of quiet or 'boat noisy' recordings. L1,L2 has the highest percentage of calls, with 50-90% of days that have calls at some point in the day.

The recordings follow the naming convention as `Location followed by deployment number`. For example,
E1 is the first deployment at Etches (Port Etches), P2 is the second deployment at Pony Cove, H1 is the first deployment at Hanning Bay and L2 is second deployment at Little Bay.

File names start with the [hydrophone](https://en.wikipedia.org/wiki/Hydrophone) serial number, followed by date and time of recording. For example: For a file named 67391498.180916010013, the recording was made with hydrophone 67391498 on 2018 September 16th at 1:03 am (and 13 seconds) UTC.

- Field Recordings NGOS

  - 2005 field recordings
  - 2006 field recordings
  to ...
  - 2018 field recordings

The field recordings have a lot of whale calls and as mentioned by Dan, we mainly have three kinds/types of killer whales

- Resident Killer Whales which feeds only on fishes. 
- Transient (or Biggs) Killer Whales which feeds on mammals.
- Offshore Killer Whales which prey mostly on shark. Their calls are very similar to Residents.

For these 14 years of field recordings, around 16 pods exist, namingly, `AD`, `AK`, `AB`, `AN`, `PWS`, `AX`, `KF`, `KW`, `AG`, `AE`, `E`, `A`, `AY`, `AF`, `AJ` and `AS`.

### Data preparation and Pre-processing

The real-world data is often noisy and requires pre-processing. To tackle this, following are some pre-processing steps to put all the samples in a standard format.

- Resampling and normalization of all the audio samples: Since the
sampling rates of all the audio samples will be different, this step will
resample all the audio files to a specific sampling frequency.
  - We resample all the audio files to a default sampling rate of 44100 Hz, unless explicitly specified by the `-r` argument.
- [Data Augmentation](https://medium.com/@alibugra/audio-data-augmentation-f26d716eee66) techniques for padding of samples.
  - This has been implemented in [preprocessing.py](https://github.com/axiom-data-science/OrcaCNN/blob/master/PreProcessing/preprocessing.py)
- Removing all the dead samples (negligible frequency)(if any) from the
dataset.
  - The `make_chunks()` method takes care of this by dividing into fixed-size chunks of size `chunkSize` specified by `-s` argument. Also, by specifying the `-m` flag as `True`, we can delete audio chunks with `db < -80`.
- Denoising: The [Spectral-Subtraction](https://doi.org/10.1109/TASSP.1979.1163209) method can be used to reduce background
noise. This method is based on spectral averaging and residual
noise reduction, widely used for enhancement of noisy speech
signals and can remove the stationary noise included in the sound.
[Sample Code](https://github.com/tracek/Ornithokrites/blob/master/noise_subtraction.py). The Wiener filter is also an option.
  - The `wavelet_denoising()` method using scikit-image effectively does the denoising.
- The spectrogram images for each audio file is prepared with Linear Short Time Fourier
Transform/Log-Mel Filter Bank features using Python. [Librosa](https://librosa.github.io/librosa/index.html) will be the
preferred choice for computing these feature representations here.
- Contrast Enhancement: [From past experience](https://github.com/jaimeps/whale-sound-classification/blob/master/7_additional_feature_design/Alternative_image_preprocessing.ipynb), Histogram Equalization
has been found as a better option than capping the extreme values to
mean ± 1.5 std.
  - This was not found necessary for the data we have at hand.
- [Focusing on Low-Frequency Spectrum](https://github.com/jaimeps/whale-sound-classification/blob/master/4_image_preprocessing_and_template_extraction_tutorial/Tutorial_image_preprocessing.ipynb): The whale calls are normally
found in the lower frequency spectrum ranging from 100Hz-200Hz.
This would allow us to look only at the specific part of the image which
will be beneficial to the CNN architecture when given as input. The
rest of the image part would roughly count as “noise” (or irrelevant)
portion for CNN.
  - This was not found necessary for the data we have at hand.
- Signal to Noise (SNR) Ratio: It is essential to have a high value of SNR for
all the audio samples. [Previous kaggle competition](http://blog.kaggle.com/2013/05/06/summary-of-the-whale-detection-competition/) have shown this as an essential factor in improving the model accuracy.
  - `wavelet_denoising()` has been implemented in [preprocess_chunk_img.py](https://github.com/axiom-data-science/OrcaCNN/blob/master/PreProcessing/preprocess_chunk_img.py) to improve the SNR ratio.
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
  - Per-Channel Energy Normalization (PCEN) has been implemented in [preprocess_chunk_img.py](https://github.com/axiom-data-science/OrcaCNN/blob/master/PreProcessing/preprocess_chunk_img.py)
- [The following paper](https://arxiv.org/pdf/1706.07156.pdf) shows that Mel-scaled STFT outperforms other methods like
Constant Q-Transform(CQT) and Constant Wavelet Transform (CWT).
**Architecturally speaking, 2D convolutions were found to perform better than
1D convolutions.**

#### Usage:
To put all the audio samples (.wav) in a standard format as described above, assuming all the [dependencies](https://github.com/axiom-data-science/OrcaCNN/blob/master/requirements.txt) are installed, simply run

```
preprocess_chunk_img.py [-h] -c CLASSPATH [-r RESAMPLING] [-s CHUNKS] [-m SILENT]

```
which will produce two folders:
- `PreProcessed_audio` with all the audio chunks of size `CHUNKS` and,
- `PreProcessed_image` with all the spectrogram images for the audio chunks in the same directory as `CLASSPATH`.