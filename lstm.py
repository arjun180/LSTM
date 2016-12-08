import keras
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

 	return n_chars,n_vocab

def create_sequence(n_chars):
	pass






if __name__ == "__main__":
	path = '/Users/arjunchakraborty/Documents/wd/LSTM/wonderland.txt'
	n_chars,n_vocab = read_file(path)
