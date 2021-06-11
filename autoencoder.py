import os
path = os.getcwd()

from os import listdir
from os.path import isfile, join 
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler

for file_name in onlyfiles:
  #Import data
  df = pd.read_csv(file_name)
  #df = df.rename(columns = {'default payment next month': 'Default'})
  print(df.shape)
  x_train =np.array(df)
  # This is the dimension of the original space
  input_dim = 82

  # This is the dimension of the latent space (encoding space)
  latent_dim = 10

  encoder = Sequential([
      #Dense(128, activation='relu', input_shape=(input_dim,)),
      #Dense(64, activation='relu'),
      #Dense(32, activation='relu'),
      #Dense(latent_dim, activation='relu')

      #Dense(64, activation='linear', input_shape=(input_dim,)),
      #Dense(32, activation='linear'),
      #Dense(latent_dim, activation='linear')

      #Dense(64, activation='relu', input_shape=(input_dim,)),
      #Dense(32, activation='relu'),
      #Dense(latent_dim, activation='relu')

      #Dense(64, activation='relu', input_shape=(input_dim,)),
      #Dense(32, activation='selu'),
      #Dense(latent_dim, activation='selu')

      Dense(latent_dim, activation=None, input_shape=(input_dim,)),
      
      #Dense(latent_dim, activation='linear', input_shape=(input_dim,)),
  ])

  decoder = Sequential([
      #Dense(64, activation='relu', input_shape=(latent_dim,)),
      #Dense(128, activation='relu'),
      #Dense(256, activation='relu'),
      #Dense(input_dim, activation=None)

      #Dense(32, activation='linear', input_shape=(latent_dim,)),
      #Dense(64, activation='linear'),
      #Dense(input_dim, activation=None)

      # Dense(32, activation='relu', input_shape=(latent_dim,)),
      # Dense(64, activation='relu'),
      # Dense(input_dim, activation=None)

      #Dense(32, activation='selu', input_shape=(latent_dim,)),
      #Dense(64, activation='selu'),
      #Dense(input_dim, activation='relu')
      
      Dense(input_dim, activation=None, input_shape=(latent_dim,)),

      #Dense(input_dim, activation='linear', input_shape=(latent_dim,)),
  ])
  autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.output))
  autoencoder.compile(loss='mse', optimizer='adam') # 'SGD'
  model_history = autoencoder.fit(x_train, x_train, epochs=60, batch_size=32, verbose=0) #5000
  #plt.plot(model_history.history["loss"])
  #plt.title("Loss vs. Epoch")
  #plt.ylabel("Loss")
  #plt.xlabel("Epoch")
  #plt.grid(True)
  encoded_x_train = encoder(x_train)
  print(encoded_x_train.shape)
  print(np.array(encoded_x_train)[0])
  stacked = pd.DataFrame(np.array(encoded_x_train))
  stacked.to_csv(file_name[:-4]+'auto.csv', index=False)