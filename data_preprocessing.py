import load_training_set
from keras.models import Model
import glob
from keras.applications.inception_v3 import InceptionV3
import time
from keras.preprocessing import image
import numpy as np
from keras.applications.inception_v3 import preprocess_input
import pickle

#-----------------------------------------------------------------------------------------------------------------
#Load model

# Get the InceptionV3 model trained on imagenet data
model = InceptionV3(weights='imagenet')
# Remove the last layer (output softmax layer) from the inception v3
model_new = Model(model.input, model.layers[-2].output)

#------------------------------------------------------------------------------------------------------------------

#Add image dataset path
images = '../Flicker8k_Dataset/'
img = glob.glob(images + '*.jpg')

#------------------------------------------------------------------------------------------------------------------

# Below file conatains the names of images to be used in train data
train_images_file = 'Dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
# Read the train image names in a set
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

# Create a list of all the training images with their full path names
train_img = []

for i in img: # img is list of full path names of all images
    if i[len(images):] in train_images: # Check if the image belongs to training set
        train_img.append(i) # Add it to the list of train images

#------------------------------------------------------------------------------------------------------------------

# Below file conatains the names of images to be used in test data
test_images_file = 'Dataset/Flickr8k_text/Flickr_8k.testImages.txt'
# Read the validation image names in a set# Read the test image names in a set
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

# Create a list of all the test images with their full path names
test_img = []

for i in img: # img is list of full path names of all images
    if i[len(images):] in test_images: # Check if the image belongs to test set
        test_img.append(i) # Add it to the list of test images        
#--------------------------------------------------------------------------------------------------------------


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
#-------------------------------------------------------------------------------------------------------------

# Function to encode a given image into a vector of size (2048, )
def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec    

#-------------------------------------------------------------------------------------------------------------

#encoding of training images.

i=0

encoding_train = {}
for img in train_img:
	i=i+1
	print(i)
	encoding_train[img[len(images):]] = encode(img)


# Save the bottleneck train features to disk
with open("Dataset/Pickle/encoded_train_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)

#--------------------------------------------------------------------------------------------------------

#encoding of testing images.

i=0
encoding_test = {}
for img in test_img:
	i=i+1
	print(i)
	encoding_test[img[len(images):]] = encode(img)


# Save the bottleneck test features to disk
with open("Dataset/Pickle/encoded_test_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)

print("done")