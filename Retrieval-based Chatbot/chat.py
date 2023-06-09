import numpy as np
import re
import random
from keras.utils import pad_sequences
from data_preprocess import df, tokenizer,lbl_enc,X
from keras.models import load_model

model=load_model('model.h5')


class ChatBot:
  #These lines define class-level variables. negative_responses is a tuple of strings representing negative responses, 
  # and exit_commands is a tuple of strings representing exit commands. This allows the code to exit
  negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")

  exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")
  

  #This method is used to initiate the chat. It prompts the user to start the chat with the chatbot, 
  # and if the response is a negative response, it exits the chat. If the response is positive, it starts the chat 
  # by calling the chat method with the response.
  def start_chat(self):
    user_response = input("Hi, I'm a chatbot mental health chatbot created by Ethan OBrien\n")
    if user_response in self.negative_responses:
      print("Ok, have a great day!\n")
      return
    
    self.chat(user_response)
  

  #This method takes an input string (reply), and continues the chat as long as make_exit returns False. 
  # The next reply from the user is obtained by calling input, and the chatbot's response is generated by calling generate_response.
  def chat(self, reply):
    while not self.make_exit(reply):
      reply = input(self.generate_answer(reply))
    

  

  #This method takes the encoded user input and uses the LTSM model to decode a response
  def generate_answer(self,pattern): 
      text = []
      txt = re.sub('[^a-zA-Z\']', ' ', pattern)
      txt = txt.lower()
      txt = txt.split()
      txt = " ".join(txt)
      text.append(txt)
      try:  
        x_test = tokenizer.texts_to_sequences(text)
        x_test = np.array(x_test).squeeze()
        x_test = pad_sequences([x_test], padding='post', maxlen=X.shape[1])
        y_pred = model.predict(x_test)
        y_pred = y_pred.argmax()
        tag = lbl_enc.inverse_transform([y_pred])[0]
        responses = df[df['tag'] == tag]['responses'].values[0]
        responses=responses[0]
        chatbot_response=responses + '\n'
          
        return chatbot_response
      except:
        return "I'm sorry, I'm afraid I do not understand\n"
  

  #This method checks if the user response contains one of the exit commands. If it does it returns true which will break the loop in
  #the chat method
  def make_exit(self, reply):
    for exit_command in self.exit_commands:
      if exit_command in reply:
        print("Ok, have a great day!")
        return True
      
    return False

chatbot=ChatBot()

chatbot.start_chat()