import numpy as np
import re
import pickle

from data_prep import pairs

# Building empty lists to hold sentences
input_docs = []
target_docs = []
# Building empty vocabulary sets
input_tokens = set()
target_tokens = set()

for line in pairs[:2000]:

  # Input and target sentences are separated by tabs
  input_doc, target_doc = line[0], line[1]

  # Appending each input sentence to input_docs
  input_docs.append(input_doc)

   #The below expression tokenizes the target_doc into a list of words and punctuation marks, 
   #and then concatenates them into a single string separated by spaces.
  target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))

  # Redefine target_doc below 
  # and append it to target_docs:
  target_doc = '<START> ' + target_doc + ' <END>'
  target_docs.append(target_doc)
  
   # Now we split up each sentence into words
  # and add each unique word to our vocabulary set
  #The below expression tokenizes the input_doc into a list of words and punctuation marks, 
  # and iterates over each token
  for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
    
    if token not in input_tokens:
      input_tokens.add(token)
  for token in target_doc.split():
    

    if token not in target_tokens:
      target_tokens.add(token)

input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))

# Create num_encoder_tokens and num_decoder_tokens:
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

#this code calculates the length of the longest sequence of tokens in the input_docs list by tokenizing 
# each input_doc into a list of tokens and finding the maximum length of these lists
max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])

#Now we need to create dictionaries of each word for the input and output. We also need 
#reverse dictionaries so that we can find the word based on the index
input_features_dict = dict(
    [(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict(
    [(token, i) for i, token in enumerate(target_tokens)])

reverse_input_features_dict = dict(
    (i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict(
    (i, token) for token, i in target_features_dict.items())

#Now we need to create numpy arrays with 0s. THe arrays will be filled with a 1 for the token that we are looking to encode and decode.
# This is because the keras model requires all words to be in one-hot encode vectors
encoder_input_data = np.zeros(
    (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

#This loop processes the input data. For each line and each token (word or punctuation) in the input document, 
    # the code finds the index of the token in the input feature dictionary (input_features_dict). 
    # This index is used to set the corresponding entry in the 3D encoder input data array (encoder_input_data) to 1. 
for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):

  for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
    # Assign 1. for the current line, timestep, & word
    # in encoder_input_data:
    encoder_input_data[line, timestep, input_features_dict[token]] = 1.
    # add in conditional for handling unknown tokens (when token is not in input features dict)

  for timestep, token in enumerate(target_doc.split()):

    decoder_input_data[line, timestep, target_features_dict[token]] = 1.
    if timestep > 0:

      decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.