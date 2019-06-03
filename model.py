import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from itertools import islice
import gc
from natsort import natsorted

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

images_path = '/home/bernhard/Documents/ml/freesound/tex_down/ch_tex'
m = len(os.listdir(images_path))
# m = 200

cols = ['fname', 'label', 'manually_verified']
df = pd.read_csv('train_post_competition.csv', usecols=cols)

# Kick out all non-manually verified files
df = df[df['manually_verified'] == 1]
df = df.drop('manually_verified', axis=1)

X_files = np.asarray(df)
Y = X_files[:m,1]
X_files = []

def load_images(path):
  image_list = natsorted(os.listdir(path))
  loaded_images = []
  for image in islice(image_list, m):
    with open(os.path.join(path, image), 'rb') as i:
      img = Image.open(i)
      data = np.asarray(img, dtype='int32')
      # data = data[:,:,:3]
      loaded_images.append(data)
  loaded_images = np.array(loaded_images)
  return loaded_images

X = load_images(images_path)
print(X.shape)

X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
print('X shape: {}'.format(X.shape))
print('Y shape: {}'.format(Y.shape))
print('Number tex: {}'.format(m))


Y_one_hot = OneHotEncoder(sparse=False)
Y = Y.reshape(Y.shape[0], 1)
Y_one_hot = Y_one_hot.fit_transform(Y)
print('Number of categories: {}'.format(len(Y_one_hot[0])))


random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X, Y_one_hot,
                                 test_size = 0.2, random_state=random_seed)

# Clear some memory
del df
X = []
gc.collect()

datagen = ImageDataGenerator(
        zoom_range = 0.2,
        width_shift_range=0.3,
        height_shift_range=0.3,
        )

datagen.fit(X_train)

model = Sequential([
  # Conv2D(32, (3,3), strides=1, input_shape=(X_train.shape[1:]), activation='relu'),
  # MaxPooling2D(pool_size=(2,2)),
  # Conv2D(64, (3,3), strides=1, activation='relu'),
  # MaxPooling2D(pool_size=(2,2)),
  # Conv2D(128, (3,3), strides=1, activation='relu'),
  # MaxPooling2D(pool_size=(2,2)),
  # Conv2D(256, (3,3), strides=1, activation='relu'),
  # MaxPooling2D(pool_size=(2,2)),
  # Dropout(0.25),
  # Flatten(),
  # Dense(128, activation='relu'),
  # Dense(64, activation='relu'),
  # Dropout(0.25),
  # Dense(41, activation='softmax'),
  Conv2D(32, (3,3), strides=1, input_shape=(X_train.shape[1:]), activation='relu'),
  Conv2D(64, (3,3), strides=1, activation='relu'),
  MaxPooling2D(pool_size=(2,2)),
  Conv2D(128, (3,3), strides=1, activation='relu'),
  Conv2D(256, (3,3), strides=1, activation='relu'),
  MaxPooling2D(pool_size=(2,2)),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(64, activation='relu'),
  Dense(41, activation='softmax'),
])

# model.summary()
# model.load_weights("sound_weights.h5")
# print('Loaded weights')

# opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=4),
             ModelCheckpoint(filepath='weights/model.h5', monitor='val_loss',
             save_best_only=True)]

history =  model.fit_generator(datagen.flow(X_train, Y_train, batch_size=42),
  epochs=90, steps_per_epoch=m/42, validation_data = (X_val,Y_val), verbose=1,
  callbacks=callbacks)


# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# model.save_weights('sound_weights.h5')
# print('Saved weights')