import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from itertools import islice
import gc
from natsort import natsorted

from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

images_path = '/home/bernhard/Documents/ml/freesound/generated_tex_gray_double_down/'
m = len(os.listdir(images_path))
# m = 2000

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
      # data = data[:,:,0]
      loaded_images.append(data)
  loaded_images = np.array(loaded_images)
  return loaded_images

X = load_images(images_path)
# Down: 300 223
# Double down: 135 100
# X = images.reshape((m, 300, 223))

X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
print('X shape: {}'.format(X.shape))
print('Y shape: {}'.format(Y.shape))
print('Number tex: {}'.format(m))

# Plot an image
# plt.imshow(X[0][:,:,0])
# plt.show()


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
        rotation_range=2,
        zoom_range = 0.2,
        width_shift_range=0.2,
        height_shift_range=0.2)

datagen.fit(X_train)

model = Sequential([
  Conv2D(32, (3,3), strides=1, input_shape=(X_train.shape[1:]), activation='relu'),
  MaxPooling2D(pool_size=(2,2)),
  Conv2D(64, (3,3), strides=1, activation='relu'),
  MaxPooling2D(pool_size=(2,2)),
  Conv2D(128, (3,3), strides=1, activation='relu'),
  MaxPooling2D(pool_size=(2,2)),
  Conv2D(256, (3,3), strides=1, activation='relu'),
  MaxPooling2D(pool_size=(2,2)),
  Dropout(0.25),
  Flatten(),
  Dense(128, activation='relu'),
  Dropout(0.5),
  Dense(41, activation='softmax'),
])

# model.summary()

# opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

# batch_size = 64
# steps_per_epoch = int(X_train.shape[0]/batch_size)
# history = model.fit(X_train, Y_train, epochs=30, steps_per_epoch=steps_per_epoch,
#                     validation_data = (X_val,Y_val), validation_steps=1, verbose=1)

history =  model.fit_generator(datagen.flow(X_train, Y_train, batch_size=42),
  epochs=30, steps_per_epoch=m/42, validation_data = (X_val,Y_val))


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

model.save_weights('sound_weights.h5')
print('Saved weights')