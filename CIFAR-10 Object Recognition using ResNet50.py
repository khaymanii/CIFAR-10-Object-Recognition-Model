

get_ipython().system('pip install kaggle')


# In[ ]:


# configuring the path of Kaggle.json file
get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[ ]:


# daatset api
get_ipython().system('kaggle competitions download -c cifar-10')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


# extracting the compessed Dataset
from zipfile import ZipFile
dataset = '/content/cifar-10.zip'

with ZipFile(dataset,'r') as zip:
  zip.extractall()
  print('The dataset is extracted')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('pip install py7zr')


# In[ ]:


import py7zr

archive = py7zr.SevenZipFile('/content/train.7z', mode='r')
archive.extractall()     #archive.extractall(path='/content/Training Data')
archive.close()


# In[ ]:


get_ipython().system('ls')


# Importing the Dependencies

# In[ ]:


import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split


# In[ ]:


filenames = os.listdir('/content/train')


# In[ ]:


type(filenames)


# In[ ]:


len(filenames)


# In[ ]:


print(filenames[0:5])
print(filenames[-5:])


# **Labels Processing**

# In[ ]:


labels_df = pd.read_csv('/content/trainLabels.csv')


# In[ ]:


labels_df.shape


# In[ ]:


labels_df.head()


# In[ ]:


labels_df[labels_df['id'] == 7796]


# In[ ]:


labels_df.head(10)


# In[ ]:


labels_df.tail(10)


# In[ ]:


labels_df['label'].value_counts()


# In[ ]:


labels_df['label']


# In[ ]:


labels_dictionary = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

labels = [labels_dictionary[i] for i in labels_df['label']]


# In[ ]:


print(labels[0:5])
print(labels[-5:])


# In[ ]:


# displaying sample image
import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread('/content/train/7796.png')
cv2_imshow(img)


# In[ ]:


# displaying sample image
import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread('/content/train/45888.png')
cv2_imshow(img)


# In[ ]:


labels_df[labels_df['id'] == 45888]


# In[ ]:


labels_df.head()


# In[ ]:


id_list = list(labels_df['id'])


# In[ ]:


print(id_list[0:5])
print(id_list[-5:])


# **Image Processing**

# In[ ]:


# convert images to numpy arrays

train_data_folder = '/content/train/'

data = []

for id in id_list:

  image = Image.open(train_data_folder + str(id) + '.png')
  image = np.array(image)
  data.append(image)


# In[ ]:


type(data)


# In[ ]:


len(data)


# In[ ]:


type(data[0])


# In[ ]:


data[0].shape


# In[ ]:


data[0]     


# In[ ]:


# convert image list and label list to numpy arrays

X = np.array(data)
Y = np.array(labels)


# In[ ]:


type(X)


# In[ ]:


print(X.shape)
print(Y.shape)


# **Train Test Split**

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[ ]:


print(X.shape, X_train.shape, X_test.shape)


# In[ ]:


# scaling the data

X_train_scaled = X_train/255

X_test_scaled = X_test/255


# In[ ]:


X_train_scaled


# In[ ]:


X_train[0]


# **Building the Neural Network**

# In[ ]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


num_of_classes = 10

# setting up the layers of Neural Network

model = keras.Sequential([
    
    keras.layers.Flatten(input_shape=(32,32,3)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_of_classes, activation='softmax')
])


# In[ ]:


# compile the neural network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])


# In[ ]:


# training the neural network
model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=10)


# **ResNet50**

# In[ ]:


from tensorflow.keras import Sequential, models, layers
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import optimizers


# In[ ]:


convolutional_base = ResNet50(weights='imagenet', include_top=False, input_shape=(256,256,3))
convolutional_base.summary()


# In[ ]:


num_of_classes = 10

model = models.Sequential()
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(convolutional_base)
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(num_of_classes, activation='softmax'))


# In[ ]:


model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='sparse_categorical_crossentropy', metrics=['acc'])


# In[ ]:


history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=10)


# In[ ]:


loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('Test Accuracy =', accuracy)


# In[ ]:


h = history

# plot the loss value
plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

# plot the accuracy value
plt.plot(h.history['acc'], label='train accuracy')
plt.plot(h.history['val_acc'], label='validation accuracy')
plt.legend()
plt.show()


# In[ ]:




