
# # Introduction
# 
# During this project I will use the Keras API to build a deep learning translator. I preprocessed the data to get one-hot encoded vectors, then train the model using LTSM from Keras, and finally test the model using various input strings
#Import all libraries
import numpy as np
import re
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
from preprocess import preprocess
from keras.callbacks import EarlyStopping
# # Preprocessing


# Importing our translations
data_path = "C:/Users/eobri/OneDrive/Desktop/ML_Projects/NLP/Translator/fra.txt"



# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:
   lines = f.read().split('\n')


input_docs,input_tokens,target_tokens,encoder_input_data, decoder_input_data, decoder_target_data, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_features_dict, target_features_dict, reverse_input_features_dict, reverse_target_features_dict= preprocess(lines)

print(list(input_features_dict.keys())[:50])

print(reverse_target_features_dict[50])


print(len(input_tokens))

# # Model Training

#the number of dimensions of the internal representation 
# of the input sequences in the encoder LSTM and decoder LSTM layers of a neural network.
latent_dim = 256
#The Batch Size is a hyperparameter of a machine learning model that defines the number of samples to work 
# through before updating the internal model parameters.
batch_size = 75
#Epoch specifies the number of times the training loop will run over the entire training data.
epochs = 10


#The below code creates an encoder network for a seq2seq model using Keras.

#This defines an input layer for the encoder with a shape of (None, num_encoder_tokens), where None represents the 
# length of the sequence and num_encoder_tokens is the number of tokens/words in the encoder vocabulary.
encoder_inputs = Input(shape=(None, num_encoder_tokens))

#This creates an LSTM layer with latent_dim number of units, and with return_state set to True, 
# which means it will return the hidden state and cell state of the LSTM layer in addition to its outputs.
#The return_state=True in the LSTM layer is important because the hidden state and cell state are needed for the decoder 
# part of the model to properly predict the output sequence. The hidden state and cell state capture the context information 
# from the input sequence that is fed into the encoder. This information is then used by the decoder to generate the target sequence.
encoder_lstm = LSTM(latent_dim, return_state=True)

#The third line applies the encoder_inputs to the encoder_lstm layer, and the outputs, state_hidden, and state_cell are assigned 
# to encoder_outputs, state_hidden, and state_cell respectively.
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)

#This defines a list called encoder_states, which includes the hidden and cell states of the LSTM layer. 
# This will be used as the initial state of the decoder network.
encoder_states = [state_hidden, state_cell]


#This code is defining and implementing an LSTM decoder in Keras, which is used in a seq2seq neural network.


#The first line creates an input layer for the decoder with shape (None, num_decoder_tokens), where "None" indicates 
# a variable-length sequence and "num_decoder_tokens" is the number of tokens in the decoder's vocabulary. The "None" in 
# the shape of the decoder_inputs layer means that the length of the input sequences can vary and it is not fixed. 
# This allows the decoder to handle inputs of different lengths, which is important because 
# the length of the source and target sentences can be very different
decoder_inputs = Input(shape=(None, num_decoder_tokens))

#The second line creates an LSTM layer with "latent_dim" units and specifies
#  to return both the sequence of outputs and the hidden and cell states.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

#The third line applies the LSTM layer to the decoder_inputs and initializes the hidden and cell states with the encoder_states.
#  It also splits the outputs and states into separate variables.
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)

#The fourth line creates a dense layer with "num_decoder_tokens" units 
# and a "softmax" activation function, which will be used to produce a probability distribution over the decoder's vocabulary.
#A dense layer is a type of layer in a neural network that has connections to all the neurons in the previous layer. 
# Each neuron in the dense layer receives input from every neuron in the previous layer. The dense layer performs a 
# matrix multiplication of the input with a weight matrix and adds a bias term to produce the output. 
# The activation function applied to the output then determines the final activation values for each neuron in the dense layer.
#The softmax function maps its input to a probability distribution over the classes, with each output value representing the predicted probability
# for each class. The softmax function normalizes the input so that the sum of all the outputs is equal to 1, which represents a valid 
# probability distribution. The class with the highest probability is chosen as the final prediction. 
decoder_dense = Dense(num_decoder_tokens, activation='softmax')

#he fifth line applies the dense layer to the decoder_outputs to obtain the final decoder output.
decoder_outputs = decoder_dense(decoder_outputs)


# Building the training model:
#By creating a model in this way, it is now possible to train the network on data by passing input 
# sequences to the encoder and decoder and using the decoder outputs as the target during training. 
# The model can be compiled, fit to data, and used to make predictions, which can be compared to the target to compute the 
# training loss and update the model's weights.
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print("Model summary:\n")
training_model.summary()
print("\n\n")


# Compile the model:
training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'],run_eagerly=True)

#Categorical cross-entropy is a loss function used for multi-class classification problems. It is used to measure the 
# difference between the predicted probability distribution and the true distribution of the target classes. 
# The output of the loss function is a scalar value that summarizes the average discrepancy between the predicted 
# class probabilities and the true class labels for the given input data.

#RMSprop (Root Mean Square Propagation) is a popular optimization algorithm used in deep learning. 
# It is a gradient descent optimization algorithm that adapts the learning rates of individual parameters 
# based on the historical gradient information. The idea behind RMSprop is to divide the learning rate for each parameter 
# by a running average of the historical magnitudes of the gradients for that parameter, effectively reducing the 
# learning rate for parameters that have consistently high gradients. This can help prevent oscillations or divergence 
# during training and lead to faster convergence. RMSprop is often used with deep neural networks, especially 
# recurrent neural networks and convolutional neural networks.

#Add early stopping to the training process:
#The early stopping callback monitors the validation loss of the model and stops the training process when the validation loss stops improving.

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

#Finally, we fit the model with the encoder and decoder inputs
training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = batch_size, 
epochs = epochs, validation_split = 0.2, callbacks=[early_stopping])

training_model.save('training_model.h5')



