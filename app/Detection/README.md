### Training and Hyperparameter Tuning

After Pre-Processing, the dataset had around `40k samples`. It was divided in the ratio `70:20:10` for the training, validation and test sets respectively.

#### Usage:

- Training the CNN:

```
orcacnn_detection.py [-h] -c CLASSPATH

```
where CLASSPATH expects a directory structure as shown below:

```
CLASSPATH
├── test
│   ├── neg
│   └── pos
├── train_orca
│   ├── neg
│   └── pos
└── val_orca
    ├── neg
    └── pos
```

#### Note: It is advisable to change the batch-size according to your system RAM.

- sklearn classification report:

```
report.py [-h] -m MODELPATH -c TESTPATH

```
where MODELPATH is the path to the saved model weights after training and TESTPATH is the directory containing your test data with the structure as shown:

```
TESTPATH
├── neg
└── pos
```

- Predicting orcas:

```
predict.py [-h] -m MODELPATH -c TESTPATH

```
where MODELPATH is the file path to the saved model weights or a directory path to multiple saved model weights after training and TESTPATH is the directory containing your pre-processed orca images. These spectrogram images are first **renamed from 0 to N-1** to help find the start and end time of orca calls. The images are then fed to the model and the predicted orca samples are saved in a new folder `pos_orca` within the same directory. A csv file named `predictions.csv` labelling which examples were predicted with high confidence and low confidence by the model(s) will also be saved to the `pos_orca` folder.

- Determine start and end time of orca in the audio sample:

```
determine_calls.py [-h] -c CLASSPATH

```
The CLASSPATH (or the folder named `pos_orca`) expects all the detected orca images from the last script `predict.py`. The start and end calls are determined as follows:
  
  - The `pos_orca` directory contains numbered samples from the last step. All these samples are 1 second long (see Pre-Processing step).
  - If there are samples numbered from 0-4 and 9-12, then there is an orca call of `5s duration from `1st-5th second` in the long sample. The second orca call starts at 10th second and lasts till 13th second of the long sample with a total duration of 4s.
  - Hence, there are total 2 orca calls of total duration `9s`.

#### Labelling Folder

- The Labelling folder was created to help label orcas from a small subset of data. Initially, the dataset was huge and was not properly labelled. To tackle this, [Jesse](https://github.com/yosoyjay), my mentor, advised me this method and thankfully I was able to label the data much faster.

The saved model inside the folder helped me with that.

