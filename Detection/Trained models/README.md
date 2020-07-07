### Saved models after Training

This folder contains some of the models in `.h5` format that I was able to save and test after developing the Detection model. Some are even checkpoints created during training.

To make good use of these models, your data must be pre-processed in the manner my model was trained on.  Below are some examples of how the spectrograms for positive (orca) and negative (non-orca/boat noises/humpbacks) look like when pre-processed using [preprocess_chunk_img.py](https://github.com/axiom-data-science/OrcaCNN/blob/master/PreProcessing/preprocess_chunk_img.py)

This is the Google Drive link with some of the [saved models](https://drive.google.com/drive/folders/16j1ceu9GB-BB1A8ImPgmGgUW8aXUvFzw?usp=sharing) since size > 100MB (git-lfs I know, but still :/)


#### Positive samples of spectrograms:

<p align = "center">
<img src = assets/positive.png>
</p>


#### Negative samples of spectrograms:

<p align = "center">
<img src = assets/negative.png>
</p>


#### NOTE: I cannot satisfyingly say which model works the best (my fault!), though I've included the model which achieved highest accuracy and also some other checkpoint models which fairly worked well on test data by accurately differentiating between orca and non-orca calls.