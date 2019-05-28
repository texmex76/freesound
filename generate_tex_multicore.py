import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math
from multiprocessing import Pool

import librosa
import librosa.display

cols = ['fname', 'label', 'manually_verified']
df = pd.read_csv('train_post_competition.csv', usecols=cols)

# Kick out all non-manually verified files
df = df[df['manually_verified'] == 1]
df = df.drop('manually_verified', axis=1)

X_files = np.asarray(df)
Y_raw = X_files[:,1]

m = X_files.shape[0]
print('Number of training examples: {}'.format(m))

# Load audio files into dataframe
path = './FSDKaggle2018.audio_train/'

X_raw = np.empty((m,1), dtype='object')
for idx in range(m):
  path_file = path + X_files[idx,0]
  _, data = wavfile.read(path_file)
  data = np.asarray(data, dtype=float)
  X_raw[idx,0] = data

fs = 44_100
# change wave data to mel-stft
def calculate_melsp(x, n_fft=1024, hop_length=128):
  stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
  log_stft = librosa.power_to_db(stft)
  melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
  return melsp

# function for creating histogram and saving it
def create_hist (alist):
  x = alist[0]
  idx = alist[1]
  a = calculate_melsp(x)
  b = librosa.display.specshow(a, sr=fs)
  plt.gcf()
  name = '/home/bernhard/Documents/ml/freesound/generated_tex/tex_' + str(idx)
  plt.savefig(name, bbox_inches = 'tight', pad_inches = -0.03)
  print('Saved spectrogram ' + str(idx))
  return

cores = 8
core_count = 0
counter = 0
a = []

for idx in range(m):
  if core_count < cores:
    a.append([X_raw[idx, 0], idx])
    core_count += 1
  if core_count == cores:
    pool = Pool(processes=8)
    _ = pool.map(create_hist, a)
    core_count = 0
    a = []