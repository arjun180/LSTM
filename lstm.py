import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

def read_file(path):
	raw_text = open(path).read()
	raw_text = raw_text.lower()
 	chars = sorted(list(set(raw_text)))
 	char_to_int = dict((c,i) for i,c in enumerate(chars))
 	n_chars = len(raw_text)
 	n_vocab = len(chars)

 	print "Total characters :",n_chars
 	print "Total vocab: ",n_vocab

 	return n_chars,n_vocab,char_to_int,raw_text

def create_sequence(n_chars,char_to_int,raw_text,seq_length=100):
	dataX =[]
	dataY =[]
	for i in range(0,n_chars-seq_length,1):
		seq_in = raw_text[i:i+seq_length]
		seq_out = raw_text[i+seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])

	n_patterns = len(dataX)

	return n_patterns,dataX,dataY


def reshape(n_patterns,dataX,dataY,seq_length=100):
	X = np.reshape(dataX,(n_patterns,seq_length,1))
	X = X/float(n_vocab)
	y = np_utils.to_categorical(dataY)
	print "The shape of input is :",X.shape

	return X,y

def create_model(X,y):
	model = Sequential()
	model.add(LSTM(256,input_shape=(X.shape[1],X.shape[2])))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1],activation='softmax'))
	model.compile(loss = 'categorical_crossentropy',optimizer='adam')
	print "Fitting model"
	model.fit(X,y,nb_epoch=10,batch_size=128)








if __name__ == "__main__":
	path = '/Users/arjunchakraborty/Documents/wd/LSTM/wonderland.txt'
	n_chars,n_vocab,char_to_int,raw_text = read_file(path)
	n_patterns,dataX ,dataY= create_sequence(n_chars,char_to_int,raw_text)
	X,y = reshape(n_patterns,dataX,dataY)
