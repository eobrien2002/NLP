import numpy as np
from tensorflow import keras
from keras.models import load_model
from preprocess import preprocess
from main import input_features_dict, target_features_dict, reverse_target_features_dict

# Load the trained model
model_path = "training_model.h5"
model = load_model(model_path)

# Define the maximum length of input and target sequences
max_encoder_seq_length = 30
max_decoder_seq_length = 50

# Define a function to encode the input sentence as a one-hot encoded vector
def encode_input_sentence(input_sentence):
    input_seq = np.zeros((1, max_encoder_seq_length, len(input_features_dict)), dtype="float32")
    for t, word in enumerate(input_sentence.split()):
        if word in input_features_dict:
            input_seq[0, t, input_features_dict[word]] = 1.0
    return input_seq

# Define a function to decode the predicted target sentence from the one-hot encoded vector
def decode_target_sentence(target_sentence):
    sampled_target_indices = [np.argmax(token) for token in target_sentence]
    sampled_target_words = [reverse_target_features_dict[index] for index in sampled_target_indices]
    return " ".join(sampled_target_words)

# Define some example input sentences to test the model
input_sentences = [
    "I am a student",
    "She is a teacher",
    "The cat is on the mat",
    "He is eating an apple",
    "What is your name?",
    "I am from France",
    "I love pizza",
    "Do you speak English?",
    "I want to go home",
    "She has a cat",
]

# Encode the input sentences and use the trained model to predict the target sentences
for input_sentence in input_sentences:
    # Encode the input sentence as a one-hot encoded vector
    input_seq = encode_input_sentence(input_sentence)
    
    # Predict the target sentence using the trained model
    target_seq = np.zeros((1, max_decoder_seq_length, len(target_features_dict)), dtype="float32")
    target_seq[0, 0, target_features_dict["<START>"]] = 1.0
    for t in range(1, max_decoder_seq_length):
        decoder_output = model.predict([input_seq, target_seq],verbose=0)
        sampled_token_index = np.argmax(decoder_output[0, t - 1, :])
        target_seq[0, t, sampled_token_index] = 1.0
        if sampled_token_index == target_features_dict["<END>"]:
            break
    
    # Decode the predicted target sentence from the one-hot encoded vector
    decoded_target_sentence = decode_target_sentence(target_seq[0])
    
    # Print the input sentence and the predicted target sentence
    print("Input sentence: ", input_sentence)
    print("Predicted target sentence: ", decoded_target_sentence)
    print()
