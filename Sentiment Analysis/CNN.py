from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Dropout,Activation, Embedding, LSTM
from data_cleaning import df
from prep_training_data import train_df, test_df
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.callbacks import EarlyStopping

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['text'])
vocab_size = len(tokenizer.word_index) + 1

# Pad the sequences
maxlen = 100
X_train = pad_sequences(tokenizer.texts_to_sequences(train_df['text']), padding='post', maxlen=maxlen)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_df['text']), padding='post', maxlen=maxlen)

# Create the embedding matrix
embedding_dim = 50
epochs=10
batch_size=32


# Create the model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(X_train, train_df['label'], epochs=epochs, verbose=False, validation_data=(X_test, test_df['label']), batch_size=batch_size, callbacks=[early_stop])

# Evaluate the model
loss, accuracy = model.evaluate(X_train, train_df['label'], verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, test_df['label'], verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# Plot the training and testing accuracy
plt.style.use('ggplot')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot the training and testing loss
plt.style.use('ggplot')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Now lets look at the results of the model on the SMS data

sentiments = []

# Iterate through the column and predict each response's sentiment, append sentiment to new list
df = df.sample(n=10000, random_state=42)
for message in df['text']:
    sentiments.append(str((model.predict(pad_sequences(tokenizer.texts_to_sequences([message]), padding='post', maxlen=maxlen),verbose=0).round())))

# add the list back to our DataFrame
df['Sentiment_CNN'] = sentiments

df['Sentiment_CNN'].value_counts()

df.Sentiment_CNN = df.Sentiment_CNN.apply(lambda x: int(float(x.strip('[]'))))

df_sent = df.groupby(['country']).Sentiment_CNN.mean().reset_index()

df_sent.sort_values(by='Sentiment_CNN') 

fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(x='country', y='Sentiment_CNN', data=df_sent,ax=ax)
plt.show()

