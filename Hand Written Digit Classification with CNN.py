'''DIGIT DETECTION MODEL'''


import tensorflow as tf
import cv2 as cv

import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print("x_train",x_train.shape)
print("y_train",y_train.shape)
print("x_test",x_test.shape)
print("y_test",y_test.shape)
import matplotlib.pyplot as plt
for i in range(3):
  plt.imshow(x_train[i],cmap="binary")
  print("The number is ",y_train[i])
  plt.show()
print(set(y_train))
np.set_printoptions(linewidth=280)
print(x_train[0])
plt.imshow(x_train[0],cmap="binary")
plt.show()
x_train=tf.keras.utils.normalize(x_train,axis=1) 
x_test=tf.keras.utils.normalize(x_test,axis=1)
print(x_train[0])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,Flatten

model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(units=128,activation=tf.nn.relu))
model.add(Dense(units=128,activation=tf.nn.relu))
model.add(Dense(units=10,activation=tf.nn.softmax))


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x_train,y_train, epochs=10)

model.evaluate(x_test,y_test)

model.save("digitdetection.model")

for i in range(0,10):
  img=cv.imread(f'{i}.png')[:,:,0]
  img=np.invert(np.array([img]))
  pred=model.predict(img)
  plt.imshow(img[0] ,cmap="binary")
  print("the number is ",np.argmax(pred))
  plt.show()




