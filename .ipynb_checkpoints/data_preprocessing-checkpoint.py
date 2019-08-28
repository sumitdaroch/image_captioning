import load_training_set
from keras.models import Model

from keras.applications.inception_v3 import InceptionV3
# Get the InceptionV3 model trained on imagenet data
model = InceptionV3(weights='imagenet')
# Remove the last layer (output softmax layer) from the inception v3
model_new = Model(model.input, model.layers[-2].output)