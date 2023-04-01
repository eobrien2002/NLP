
from keras.models import Sequential, save_model
from keras.layers import Input, Embedding, LSTM, LayerNormalization, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from data_preprocess import vacab_size, X,y
import numpy as np

model = Sequential()
model.add(Input(shape=(X.shape[1])))
model.add(Embedding(input_dim=vacab_size+1, output_dim=100, mask_zero=True))
model.add(LSTM(32, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(32, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(32))
model.add(LayerNormalization())
model.add(Dense(128, activation="relu"))
model.add(LayerNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(LayerNormalization())
model.add(Dropout(0.2))
model.add(Dense(len(np.unique(y)), activation="softmax"))
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(x=X,y=y,batch_size=10,callbacks=[EarlyStopping(monitor='accuracy', patience=3)],epochs=50)

save_model(model,'model.h5')