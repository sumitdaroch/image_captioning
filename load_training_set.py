

filename = 'Dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
file = open(filename, 'r')
doc = file.read()

# print(doc)

dataset = list()
# process line by line
for line in doc.split('\n'):
	# skip empty lines
	if len(line) < 1:
		continue
	# get the image identifier
	identifier = line.split('.')[0]
	dataset.append(identifier)

# print(dataset)

train=dataset

#---------------------------------------------------------------------------------------------------------------

filename = 'descriptions.txt'
file = open(filename, 'r')
doc = file.read()

descriptions = dict()
for line in doc.split('\n'):
	# split line by white space
	tokens = line.split()
	# split id from description
	image_id, image_desc = tokens[0], tokens[1:]
	# skip images not in the set
	if image_id in dataset:
		# create list
		if image_id not in descriptions:
			descriptions[image_id] = list()
		# wrap description in tokens
		desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
		# store
		descriptions[image_id].append(desc)

train_descriptions=descriptions

#print(train_descriptions['3241965735_8742782a70'])		
#-------------------------------------------------------------------------------------------------------------        