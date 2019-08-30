from data_generator import model,max_length
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from data_preprocessing_2 import wordtoix,ixtoword
from keras.preprocessing import image
import matplotlib.pyplot as plt
from data_preprocessing import model_new
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input





model.load_weights("model_weights/model_30.h5")
images = '../Flicker8k_Dataset/'

with open("Dataset/Pickle/encoded_test_images.pkl", "rb") as encoded_pickle:
    encoding_test = pickle.load(encoded_pickle)

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x      

def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec 

  

image1 = (encode('./cow.jpeg'))
image2 = np.array(image1)
image2= image2.reshape((1,2048))

print(image2)

# pic = list(encoding_test.keys())[6]
# image = encoding_test[pic].reshape((1,2048))
# print(pic)
# print(image)

print("Greedy:",greedySearch(image2))


