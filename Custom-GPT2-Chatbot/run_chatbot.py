
import torch
from transformers import GPT2Tokenizer, GPT2Model
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F


saved_data = torch.load('gpt2_model.pth')
model = GPT2Model.from_pretrained(saved_data['model_name'], state_dict=saved_data['model_state_dict'])


tokenizer = saved_data['tokenizer']
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re

saved_data = torch.load('gpt2_model.pth')
model = GPT2LMHeadModel.from_pretrained(saved_data['model_name'], state_dict=saved_data['model_state_dict'])
tokenizer = saved_data['tokenizer']

model.eval()


exit_commands = ['exit', 'no', 'stop', 'quit']
checkpoint=0
# Get input from user
print('Hello. I am a generative hockey chatbot. I will finish your sentences for you. Try inputs like: Sidney Crosby is\n')
while True:
    
    prompt = input("You: ")

    
    if prompt.lower() in exit_commands:
        print("Bot: Goodbye!")
        break

   
        # Encode the prompt
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    attention_mask = torch.ones(encoded_prompt.shape, dtype=torch.long)
    pad_token_id = tokenizer.eos_token_id

        # Generate a response
    output = model.generate(encoded_prompt, temperature=0.7, do_sample=True, attention_mask=attention_mask, pad_token_id=pad_token_id,max_length=70)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.rsplit(".", 1)[0]

    tokens = response.split()
    num_repeats = 0
    for i in range(len(tokens) - 1):
        if tokens[i] == tokens[i+1]:
            num_repeats += 1

    # Cut off the response if it contains too many repexit
  
    max_repeats = 3
    if num_repeats > max_repeats:
        pattern = r'\b(' + re.escape(tokens[i]) + r')\b(\W+\1\b)+'
        response = re.sub(pattern, r'\1', response)
        response = response[:response.rfind('.')+1]
    
    response = response.replace('"', '')

    if '\n\n' in response:
        empty_line_index = response.index('\n\n')
        response = response[:empty_line_index]
        response = response.strip()
        response += '.'
        
    print("Bot:", response)
  

    

