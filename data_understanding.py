import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

#load text file

filename = "Dataset/Flickr8k_text/Flickr8k.token.txt"
file = open(filename, 'r')
doc = file.read()

#doc1="Hello.jpg My name is sumit \n Hello.jpg akdhskjasdh"
 #print(doc)

#----------------------------------------------------------------------------------------------------------------------

#creating a dictionary contain image name as key and caption as value.

descriptions = dict()
for line in doc.split('\n'):
    # split line by white space
    tokens = line.split()
    if len(line) < 2:
    	continue
    # take the first token as image id, the rest as description
    image_id, image_desc = tokens[0], tokens[1:]
    
    # extract filename from image id
    image_id = image_id.split('.')[0]
    
    # convert description tokens back to string
    image_desc = ' '.join(image_desc)
    if image_id not in descriptions:
        descriptions[image_id] = list()
    descriptions[image_id].append(image_desc)
#-------------------------------------------------------------------------------------------------------------