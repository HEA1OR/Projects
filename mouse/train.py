import numpy as np
import os
import cv2
import pyautogui
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *


# Note that there are actually 2560x1440 pixels on my screen
# I am simply recording one less, so that when we divide by these
# numbers, we will normalize between 0 and 1. Note that mouse
# coordinates are reported starting at (0, 0), not (1, 1)
width, height = 2559, 1439

filepaths = os.listdir(src)
X, Y = [], []
for filepath in filepaths:
  x, y, _ = filepath.split(' ')
  x = float(x) / width
  y = float(y) / height
  X.append(cv2.imread(root + filepath))
  Y.append([x, y])
X = np.array(X) / 255.0
Y = np.array(Y)
print (X.shape, Y.shape)


model = Sequential()
model.add(Conv2D(32, 3, 2, activation = 'relu', input_shape = (12, 44, 3)))
model.add(Conv2D(64, 2, 2, activation = 'relu'))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(2, activation = 'sigmoid'))
model.compile(optimizer = "adam", loss = "mean_squared_error")
model.summary()


epochs = 200
for epoch in range(epochs):
  model.fit(X, Y, batch_size = 32)

  while True:
    eyes = scan()
    if not eyes is None:
      eyes = np.expand_dims(eyes / 255.0, axis=0)
      x, y = model.predict(eyes)[0]
      pyautogui.moveTo(x * width, y * height)