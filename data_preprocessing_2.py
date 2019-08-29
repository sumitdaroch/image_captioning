import pickle
from load_training_set import train_descriptions
#store the encoded train code in train_feature

train_features = open("Dataset/Pickle/encoded_train_images.pkl", "rb")

#---------------------------------------------------------------------------------------------------------

#Creating list of taining set

all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)

#print(len(all_train_captions))

#---------------------------------------------------------------------------------------------------------
#consider only those words which occur at least 10 times

word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

#---------------------------------------------------------------------------------------------------------

#for getting value from index and index from value

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1
vocab_size = len(ixtoword) + 1     

# ----------------------------------------------------------------------------------------------------------    


# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# determine the maximum sequence length
max_length = max_length(train_descriptions)

# ----------------------------------------------------------------------------------------------------------    
