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
# m = 25
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

def save_melsp(x, name, n_fft=1024, hop_length=128):
  stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
  log_stft = librosa.power_to_db(stft)
  melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
  b = librosa.display.specshow(melsp, sr=sr)
  plt.gcf()

  directory = base_dir + 'tex/melsp_tex/' + name
  plt.savefig(directory, bbox_inches = 'tight', pad_inches = -0.03)
  print('Saved melsp ' + name)

  return

def save_har_per(x, name):
  y_harmonic, y_percussive = librosa.effects.hpss(x)
  S_harmonic   = librosa.feature.melspectrogram(y_harmonic, sr=sr)
  S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)
  log_Sh = librosa.power_to_db(S_harmonic, ref=np.max)
  log_Sp = librosa.power_to_db(S_percussive, ref=np.max)
  librosa.display.specshow(log_Sh, sr=sr)

  directory = base_dir + 'tex/sh_tex/' + name
  plt.savefig(directory, bbox_inches = 'tight', pad_inches = -0.03)
  print('Saved sh ' + name)

  librosa.display.specshow(log_Sp, sr=sr)

  directory = base_dir + 'tex/sp_tex/' + name
  plt.savefig(directory, bbox_inches = 'tight', pad_inches = -0.03)
  print('Saved sp ' + name)

  return

def save_chromo(x, name):
  y_harmonic, _ = librosa.effects.hpss(x)
  C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
  librosa.display.specshow(C, sr=sr, vmin=0, vmax=1)
  
  directory = base_dir + 'tex/ch_tex/' + name
  plt.savefig(directory, bbox_inches = 'tight', pad_inches = -0.03)
  print('Saved ch ' + name)
  return

def save_mfcc(x, name):
  S = librosa.feature.melspectrogram(x, sr=sr, n_mels=128)
  log_S = librosa.power_to_db(S, ref=np.max)
  mfcc        = librosa.feature.mfcc(S=log_S, n_mfcc=13)
  librosa.display.specshow(mfcc)
  
  directory = base_dir + 'tex/mfcc_tex/' + name
  plt.savefig(directory, bbox_inches = 'tight', pad_inches = -0.03)
  print('Saved mfcc ' + name)
  return

base_dir = '/home/bernhard/Documents/ml/freesound/'

def generate_tex(alist):
  x = alist[0]
  idx = alist[1]
  name = f'{idx:04}'
  save_melsp(x, name)
  save_har_per(x, name)
  save_chromo(x, name)
  save_mfcc(x, name)

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
    _ = pool.map(generate_tex, a)
    core_count = 0
    a = []