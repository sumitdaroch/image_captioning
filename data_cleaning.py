import data_understanding
from data_understanding import descriptions
import string

# print(descriptions['1000268201_693b08cb0e'])
#--------------------------------------------------------------------------------------------------------------
#removing punctuation and lowercase the letter

table = str.maketrans('', '', string.punctuation)
for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

#print(descriptions['1000268201_693b08cb0e'])
#-------------------------------------------------------------------------------------------------------------

#All the unique words present in the description

all_desc = set()
for key in descriptions.keys():
	[all_desc.update(d.split()) for d in descriptions[key]]

vocabulary=all_desc		

#print('Original Vocabulary Size: %d' % len(vocabulary))-->8763

#--------------------------------------------------------------------------------------------------------------

## save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

save_descriptions(descriptions, 'descriptions.txt')