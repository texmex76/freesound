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

# m = X_files.shape[0]
m = 200
print('Number of training examples: {}'.format(m))

# Load audio files into dataframe
path = './FSDKaggle2018.audio_train/'

X_raw = np.empty((m,1), dtype='object')
for idx in range(m):
  path_file = path + X_files[idx,0]
  _, data = wavfile.read(path_file)
  data = np.asarray(data, dtype=float)
  X_raw[idx,0] = data

sr = 44_100
# change wave data to mel-stft
def show_melsp(x, n_fft=1024, hop_length=128):
  stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
  log_stft = librosa.power_to_db(stft)
  melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
  b = librosa.display.specshow(melsp, sr=sr)
  plt.gcf()
  plt.show()
  return

def show_har_per(x):
  y_harmonic, y_percussive = librosa.effects.hpss(x)
  S_harmonic   = librosa.feature.melspectrogram(y_harmonic, sr=sr)
  S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)
  log_Sh = librosa.power_to_db(S_harmonic, ref=np.max)
  log_Sp = librosa.power_to_db(S_percussive, ref=np.max)
  librosa.display.specshow(log_Sh, sr=sr, y_axis='mel')
  plt.show()
  librosa.display.specshow(log_Sp, sr=sr, x_axis='time', y_axis='mel')
  plt.show()
  return

def show_chromo(x):
  y_harmonic, _ = librosa.effects.hpss(x)
  C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
  librosa.display.specshow(C, sr=sr, vmin=0, vmax=1)
  plt.show()
  return

def show_mfcc(x):
  S = librosa.feature.melspectrogram(x, sr=sr, n_mels=128)
  log_S = librosa.power_to_db(S, ref=np.max)
  mfcc        = librosa.feature.mfcc(S=log_S, n_mfcc=13)
  # delta_mfcc  = librosa.feature.delta(mfcc)
  # delta2_mfcc = librosa.feature.delta(mfcc, order=2)
  librosa.display.specshow(mfcc)
  plt.show()
  # librosa.display.specshow(delta_mfcc)
  # plt.show()
  # librosa.display.specshow(delta2_mfcc)
  # plt.show()
  return

for idx in range(30, 35):
  show_melsp(X_raw[idx,0])
  show_har_per(X_raw[idx,0])
  show_chromo(X_raw[idx,0])
  show_mfcc(X_raw[idx,0])