### Training and Hyperparameter Tuning

For the classification of pods, the dataset had `6371 samples`. It was divided in the ratio `70:20:10` for the training, validation and test sets respectively.

#### Usage:

- Training the CNN:

```
orcacnn_classification.py [-h] -c CLASSPATH

```
where CLASSPATH expects a directory structure as shown below:

```
CLASSPATH
├── test
│   ├── AB
│   ├── AB25
│   ├── AD11
│   ├── AD16
│   ├── AD5
│   ├── AD8
│   ├── AE
│   ├── AF22
│   ├── AF4
│   ├── AG
│   ├── AI
│   ├── AJ
│   ├── AJ14
│   ├── AK
│   ├── AK1
│   ├── AK2
│   ├── AS
│   ├── AX32
│   ├── AX48
│   └── AY
├── train
│   ├── AB
│   ├── AB25
│   ├── AD11
│   ├── AD16
│   ├── AD5
│   ├── AD8
│   ├── AE
│   ├── AF22
│   ├── AF4
│   ├── AG
│   ├── AI
│   ├── AJ
│   ├── AJ14
│   ├── AK
│   ├── AK1
│   ├── AK2
│   ├── AS
│   ├── AX32
│   ├── AX48
│   └── AY
└── val
    ├── AB
    ├── AB25
    ├── AD11
    ├── AD16
    ├── AD5
    ├── AD8
    ├── AE
    ├── AF22
    ├── AF4
    ├── AG
    ├── AI
    ├── AJ
    ├── AJ14
    ├── AK
    ├── AK1
    ├── AK2
    ├── AS
    ├── AX32
    ├── AX48
    └── AY

```

#### Note: It is advisable to change the batch-size according to your system RAM. The dataset has been developed with around 300 images in each of the 20 pods (AJ22 and AN10 has 0 images)



- sklearn classification report:

```
report.py [-h] -m MODELPATH -c TESTPATH

```
where MODELPATH is the path to the saved model weights after training and TESTPATH is the directory containing your test data with the structure as shown:

```
TESTPATH
	├── AB
	├── AB25
	├── AD11
	├── AD16
	├── AD5
	├── AD8
	├── AE
	├── AF22
	├── AF4
	├── AG
	├── AI
	├── AJ
	├── AJ14
	├── AK
	├── AK1
	├── AK2
	├── AS
	├── AX32
	├── AX48
	└── AY
```
