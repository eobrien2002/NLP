from preprocessing import input_features_dict, target_features_dict, reverse_input_features_dict, reverse_target_features_dict, max_decoder_seq_length, input_docs, target_docs, input_tokens, target_tokens, max_encoder_seq_length
from training_model import decoder_inputs, decoder_lstm, decoder_dense, encoder_input_data, num_decoder_tokens, num_encoder_tokens

from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
import numpy as np
import re

training_model = load_model('training_model.h5')
###### because we're working with a saved model

#this code is extracting the inputs, outputs, and hidden states of the encoder part of a pre-trained neural machine translation model.

#This is the first input layer of the model, which corresponds to the source language sequence in a neural machine translation model.
encoder_inputs = training_model.input[0]

#These three variables correspond to the output and hidden states of the encoder part of the model. 
# encoder_outputs is the final output of the encoder, which is used as the input for the decoder part of the model. 
# state_h_enc and state_c_enc are the hidden states of the encoder, which are used to initialize the hidden states of 
# the decoder part of the model.
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output

#This is a list that combines the hidden states state_h_enc and state_c_enc of the encoder. 
# These hidden states capture the context of the source language sequence and are used to initialize the hidden states of the decoder.
encoder_states = [state_h_enc, state_c_enc]

#Now we build the model:
encoder_model = Model(encoder_inputs, encoder_states)


#This code defines a new model, decoder_model, which is the decoder part of a neural machine translation model. 
# The code is defining the inputs and outputs of the decoder and how they are related.

#These are two input layers, each with a shape of (latent_dim,). latent_dim is a hyperparameter that defines the number of 
# dimensions in the hidden state of the decoder. The two input layers represent the initial hidden states of the decoder.
latent_dim = 256
decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))

#This is a list that combines the two input layers, decoder_state_input_hidden and decoder_state_input_cell.
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

#These are the output and hidden states of the decoder. decoder_lstm is an LSTM layer that takes the decoder_inputs and the 
# initial hidden states decoder_states_inputs as inputs, and produces the output and hidden states decoder_outputs, state_hidden, and state_cell.
decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

#This is a list that combines the hidden states state_hidden and state_cell of the decoder.
decoder_states = [state_hidden, state_cell]

#This line applies a dense layer, decoder_dense, to the output of the decoder, decoder_outputs.
decoder_outputs = decoder_dense(decoder_outputs)

#This line creates the decoder_model using the Model class from the Keras library. The first argument is a list that 
# combines the decoder_inputs and decoder_states_inputs, which are the inputs of the decoder. The second argument is a list 
# that combines the decoder_outputs and decoder_states, which are the outputs of the decoder.
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def decode_sequence(test_input):
  # Encode the input as state vectors.
  states_value = encoder_model.predict(test_input)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1, num_decoder_tokens))
  # Populate the first token of target sequence with the start token.
  target_seq[0, 0, target_features_dict['<START>']] = 1.

  # Sampling loop for a batch of sequences
  # (to simplify, here we assume a batch of size 1).
  decoded_sentence = ''

  stop_condition = False
  while not stop_condition:
    # Run the decoder model to get possible 
    # output tokens (with probabilities) & states
    output_tokens, hidden_state, cell_state = decoder_model.predict(
      [target_seq] + states_value)

    # Choose token with highest probability
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_token = reverse_target_features_dict[sampled_token_index]
    decoded_sentence += " " + sampled_token

    # Exit condition: either hit max length
    # or find stop token.
    if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
      stop_condition = True

    # Update the target sequence (of length 1).
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, sampled_token_index] = 1.

    # Update states
    states_value = [hidden_state, cell_state]

  return decoded_sentence