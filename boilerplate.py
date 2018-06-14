from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from sklearn.utils import shuffle
#####################################################################

"""
def load_data(path)
def to_one_hot(label_data,num_class)
def Network(input)
def loss_function(logit,label)
def Accuracy_Evaluate(prediction,Label)
def main(train_data,train_label,no_of_epochs=15000,batchsize=32)

from tensorflow.python.framework import ops
ops.reset_default_graph()
global sess
config = tf.ConfigProto()
sess = tf.Session(config = config)
graph = tf.get_default_graph()

train_data, train_label = load_data('/root/Desktop/vgg16/data')

plt.imshow(train_data[110,:,:,0])
plt.show()
print(np.argmax(train_label[110]))

main(train_data, train_label)

"""