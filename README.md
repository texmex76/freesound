# Audio classification with the freesound dataset

My take on this [Kaggle challenge](https://www.kaggle.com/c/freesound-audio-tagging/data), described in [this paper](https://arxiv.org/abs/1807.09902). The task is to classify audio. There are 41 categories.

## Quick outline

The dataset consists of a part labeled by humans and a part labeled by a machine. Here I only used the manually labeled part of the dataset for training and validation. The main idea of the model I used is to generate a spectrogram (i.e. an image) and feed that to a CNN. Initlally, that got me ~72% validation accuracy. With some more feature engineering and ensembling, I finally obtained ~82% validation accuracy.

## The files

- `explore.py`: Visualize the dataset, generate different kinds of spectrograms
- `generate_tex_multicore.py`: Generate training examples. For training, I changed the pictures to 256 grayscale and scaled them down to 135x100.
- `model.py`: Set up the model and train.
- `Ensembling.ipynb`: Average the prediction of all 5 models.