# NLP
## Sentiment Analysis
In this directory, I built a few different scripts to train sentiment binary classification models. I trained a Naive Bayes, CNN, LTSM, and Custom BERT model. The fine-tuned BERT model performed the best as the original architecture is trained on the downstream sentinel task. There is also a preprocessing script to clean general social media text.

## Generative Chatbot
This directory contains all files for a mental health generative chatbot. It is a seq2seq LTSM model trained on conversations between doctors and patients. The directory contains all files needed to run the chatbot and it can easily be connected to a front end through a flask API.

## Machine Translator
I built a seq2seq LTSM machine translator for the french to the English language. The directory contains an inference.py file where you can test out the model.

## Retrieval-based Chatbot
During this project, I wanted to build a more reliable chatbot. I trained a bi-directional LTSM model to map user inputs to intent tags to retrieve pred-defined responses. The intens.json file can be modified to expand the functionality of the chatbot and apply it to other applications.

## Custom GPT2
In this directory, I fine-tuned a CPT2 model from Hugging Face on interview data with NHL hockey players and coaches. There is a notebook that shows how to train a GPT2 model for downstream tasks, including all preprocessing and tokenizing. The run_chatbot.py script allows the user input to be completed by the model as if it was an NHL hockey interview.
