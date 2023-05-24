import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from keras_sequential_ascii import keras2ascii


class NN_seq:

    def __init__(self, nodes, activation, input):
        self.model = tf.keras.models.Sequential()
        self.dropoutval = 0.6
        if activation == 'LR':
            self.model.add(tf.keras.layers.Dense(nodes, activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_shape=(input, )))

    def add_dense(self, nodes_layers, activation):
            if activation == 'LR':
                i=1
                for num in nodes_layers:
                    if i%2==0 and num>64:
                        self.model.add(tf.keras.layers.Dropout(self.dropoutval))
                    self.model.add(tf.keras.layers.Dense(num, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
                    i+=1
    def add_output_layer(self, nodes):
        self.model.add(tf.keras.layers.Dense(nodes))



# m = NN_seq(1028, 'LR', 14)
# m.add_dense([1028,1028,512,512,256, 256,128,128,64,64,32,16], 'LR')
# m.add_output_layer(1)
# print(keras2ascii(m.model))