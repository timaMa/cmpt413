# Code adapted from original code by Robert Guthrie

import os, sys, optparse, gzip, re, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import string

def read_conll(handle, input_idx=0, label_idx=2):
	conll_data = []
	contents = re.sub(r'\n\s*\n', r'\n\n', handle.read())
	contents = contents.rstrip()
	for sent_string in contents.split('\n\n'):
		annotations = list(zip(*[ word_string.split() for word_string in sent_string.split('\n') ]))
		assert(input_idx < len(annotations))
		if label_idx < 0:
			conll_data.append( annotations[input_idx] )
			logging.info("CoNLL: {}".format( " ".join(annotations[input_idx])))
		else:
			assert(label_idx < len(annotations))
			conll_data.append(( annotations[input_idx], annotations[label_idx] ))
			logging.info("CoNLL: {} ||| {}".format( " ".join(annotations[input_idx]), " ".join(annotations[label_idx])))
	return conll_data

def prepare_sequence(seq, to_ix, unk):
	idxs = []
	# print(to_ixs)
	# exit(0)
	if unk not in to_ix:
		idxs = [to_ix[w] for w in seq]
	else:
		idxs = [to_ix[w] for w in map(lambda w: unk if w not in to_ix else w, seq)]
	return torch.tensor(idxs, dtype=torch.long)

def prepare_char(seq, to_ix):
	one_hot = []
	length = len(string.printable)
	for w in seq:
		start = torch.zeros(length)
		mid = torch.zeros(length)
		end = torch.zeros(length)
		if len(w) == 1:
			start[to_ix[w]] += 1
		elif len(w) == 2:
			start[to_ix[w[0]]] += 1
			end[to_ix[w[-1]]] += 1
		else: # >= 3 letters
			start[to_ix[w[0]]] += 1
			end[to_ix[w[-1]]] += 1
			for l in w[1:-1]:
				mid[to_ix[l]] += 1
		vector = torch.cat((start, mid, end), dim=-1)
		one_hot.append(vector)
	one_hot = torch.stack(one_hot)
	return torch.tensor(one_hot, dtype=torch.long)



class LSTMTaggerModel(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
		torch.manual_seed(1)
		super(LSTMTaggerModel, self).__init__()
		self.hidden_dim = hidden_dim

		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		# The LSTM takes word embeddings as inputs, and outputs hidden states
		# with dimensionality hidden_dim.
		self.lstm = nn.LSTM(embedding_dim + 64, hidden_dim, bidirectional=False)
		# self.lstm = nn.LSTM(embedding_dim + 300, hidden_dim, bidirectional=False)

		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

	def forward(self, sentence, chars):
		embeds = self.word_embeddings(sentence)
		embeds = torch.cat((embeds, chars), dim=1)
		lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
		tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
		tag_scores = F.log_softmax(tag_space, dim=1)
		return tag_scores

class SecondRNN(nn.Module):

	def __init__(self, input_dim = 300, hidden_dim = 128, output_dim = 64, target_size = 22):
		torch.manual_seed(1)
		super(SecondRNN, self).__init__()
		self.hidden_dim = hidden_dim

		self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=False)
		
		# The linear layer that maps from hidden state space to tag space
		self.hidden1 = nn.Linear(hidden_dim, output_dim)
		self.hidden2tag = nn.Linear(output_dim, target_size)

	def forward(self, chars):
		# print(chars.view(len(sentence), 1, -1).type())
		chars = chars.float()
		length = chars.shape[0]
		lstm_out, _ = self.lstm(chars.view(length, 1, -1))
		hidden_out = self.hidden1(lstm_out.view(length, -1))
		tag_space = self.hidden2tag(hidden_out.view(length, -1))
		tag_scores = F.log_softmax(tag_space, dim=1)
		return tag_scores

class LSTMTagger:

	def __init__(self, trainfile, modelfile, rnnfile, modelsuffix, unk="[UNK]", epochs=10, embedding_dim=128, hidden_dim=64):
		self.unk = unk
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.epochs = epochs
		self.modelfile = modelfile
		self.rnnfile = rnnfile
		self.modelsuffix = modelsuffix
		self.training_data = []
		if trainfile[-3:] == '.gz':
			with gzip.open(trainfile, 'rt') as f:
				self.training_data = read_conll(f)
		else:
			with open(trainfile, 'r') as f:
				self.training_data = read_conll(f)

		self.word_to_ix = {} # replaces words with an index (one-hot vector)
		self.tag_to_ix = {} # replace output labels / tags with an index
		self.ix_to_tag = [] # during inference we produce tag indices so we have to map it back to a tag
		
		self.char_to_ix = {}
		for c in string.printable:
			self.char_to_ix[c] = len(self.char_to_ix)

		for sent, tags in self.training_data:
			for word in sent:
				if word not in self.word_to_ix:
					self.word_to_ix[word] = len(self.word_to_ix)
			for tag in tags:
				if tag not in self.tag_to_ix:
					self.tag_to_ix[tag] = len(self.tag_to_ix)
					self.ix_to_tag.append(tag)

		logging.info("word_to_ix:", self.word_to_ix)
		logging.info("tag_to_ix:", self.tag_to_ix)
		logging.info("ix_to_tag:", self.ix_to_tag)

		self.rnn = SecondRNN(target_size = len(self.tag_to_ix))
		self.model = LSTMTaggerModel(self.embedding_dim, self.hidden_dim, len(self.word_to_ix), len(self.tag_to_ix))
		self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
		self.optimizer_rnn = optim.SGD(self.rnn.parameters(), lr=0.02)

	def argmax(self, seq):
		output = []
		with torch.no_grad():
			inputs = prepare_sequence(seq, self.word_to_ix, self.unk)
			char_in = prepare_char(seq, self.char_to_ix)
			# tag_scores = self.model(inputs, chars_in)
			rnn_scores = self.rnn(char_in)
			rnn_out = char_in.float()
			layers = list(self.rnn.children())[:-1]
			rnn_out, _ = layers[0](rnn_out.view(char_in.shape[0], 1, -1))
			rnn_out = layers[1](rnn_out.view(char_in.shape[0], -1))
			# print(rnn_out.shape)
			tag_scores = self.model(inputs, rnn_out)
			for i in range(len(inputs)):
				output.append(self.ix_to_tag[int(tag_scores[i].argmax(dim=0))])
		return output

	def train_RNN(self):
		loss_function = nn.NLLLoss()

		# Train a second RNN
		self.rnn.train()
		loss = float("inf")
		for epoch in range(20):
			for sentence, tags in tqdm.tqdm(self.training_data):
				# Step 1. Remember that Pytorch accumulates gradients.
				# We need to clear them out before each instance
				self.rnn.zero_grad()

				# Step 2. Get our inputs ready for the network, that is, turn them into
				# Tensors of word indices.
				sentence_in = prepare_sequence(sentence, self.word_to_ix, self.unk)
				targets = prepare_sequence(tags, self.tag_to_ix, self.unk)
				char_in = prepare_char(sentence, self.char_to_ix)
				# Step 3. Run our forward pass.
				rnn_scores = self.rnn(char_in)
				# rnn_out = char_in.float()
				# layers = list(self.rnn.children())[:-1]
				# rnn_out, _ = layers[0](rnn_out.view(char_in.shape[0], 1, -1))
				# rnn_out = layers[1](rnn_out.view(char_in.shape[0], -1))
				# # print(rnn_out.shape)
				# tag_scores = self.model(sentence_in, rnn_out)

				# Step 4. Compute the loss, gradients, and update the parameters by
				#  calling optimizer.step()
				loss = loss_function(rnn_scores, targets)
				loss.backward()
				self.optimizer_rnn.step()

			if epoch == self.epochs-1:
				epoch_str = '' # last epoch so do not use epoch number in model filename
			else:
				epoch_str = str(epoch)
			savernnfile = self.rnnfile + epoch_str + self.modelsuffix
			print("saving model file: {}".format(savernnfile), file=sys.stderr)
			torch.save({
						'epoch': epoch,
						'model_state_dict': self.rnn.state_dict(),
						'optimizer_state_dict': self.optimizer_rnn.state_dict()
					}, savernnfile)

	def train(self):
		loss_function = nn.NLLLoss()

		# Train a second RNN
		if not os.path.isfile(self.rnnfile + self.modelsuffix): 
			print("Train RNN model")
			self.train_RNN()
		else:
			print("Load RNN model")
			saved_rnn = torch.load(self.rnnfile + self.modelsuffix)
			self.rnn.load_state_dict(saved_rnn['model_state_dict'])

		self.model.train()
		self.rnn.eval()
		loss = float("inf")
		for epoch in range(self.epochs):
			for sentence, tags in tqdm.tqdm(self.training_data):
				# Step 1. Remember that Pytorch accumulates gradients.
				# We need to clear them out before each instance
				self.model.zero_grad()
				# Step 2. Get our inputs ready for the network, that is, turn them into
				# Tensors of word indices.
				sentence_in = prepare_sequence(sentence, self.word_to_ix, self.unk)
				targets = prepare_sequence(tags, self.tag_to_ix, self.unk)
				char_in = prepare_char(sentence, self.char_to_ix)
				# Step 3. Run our forward pass.
				# rnn_scores = self.rnn(char_in)
				rnn_out = char_in.float()
				layers = list(self.rnn.children())[:-1]
				rnn_out, _ = layers[0](rnn_out.view(char_in.shape[0], 1, -1))
				rnn_out = layers[1](rnn_out.view(char_in.shape[0], -1))
				# print(rnn_out.shape)
				tag_scores = self.model(sentence_in, rnn_out)

				# Step 4. Compute the loss, gradients, and update the parameters by
				#  calling optimizer.step()
				loss = loss_function(tag_scores, targets)
				loss.backward()
				self.optimizer.step()

			if epoch == self.epochs-1:
				epoch_str = '' # last epoch so do not use epoch number in model filename
			else:
				epoch_str = str(epoch)
			savefile = self.modelfile + epoch_str + self.modelsuffix
			print("saving model file: {}".format(savefile), file=sys.stderr)
			torch.save({
						'epoch': epoch,
						'model_state_dict': self.model.state_dict(),
						'optimizer_state_dict': self.optimizer.state_dict(),
						'loss': loss,
						'unk': self.unk,
						'word_to_ix': self.word_to_ix,
						'tag_to_ix': self.tag_to_ix,
						'ix_to_tag': self.ix_to_tag,
					}, savefile)

	def decode(self, inputfile):
		if inputfile[-3:] == '.gz':
			with gzip.open(inputfile, 'rt') as f:
				input_data = read_conll(f, input_idx=0, label_idx=-1)
		else:
			with open(inputfile, 'r') as f:
				input_data = read_conll(f, input_idx=0, label_idx=-1)

		if not os.path.isfile(self.modelfile + self.modelsuffix):
			raise IOError("Error: missing model file {}".format(self.modelfile + self.modelsuffix))

		saved_model = torch.load(self.modelfile + self.modelsuffix)
		saved_rnn = torch.load(self.rnnfile + self.modelsuffix)
		self.model.load_state_dict(saved_model['model_state_dict'])
		self.rnn.load_state_dict(saved_rnn['model_state_dict'])
		self.optimizer.load_state_dict(saved_model['optimizer_state_dict'])
		self.optimizer_rnn.load_state_dict(saved_rnn['optimizer_state_dict'])
		epoch = saved_model['epoch']
		loss = saved_model['loss']
		self.unk = saved_model['unk']
		self.word_to_ix = saved_model['word_to_ix']
		self.tag_to_ix = saved_model['tag_to_ix']
		self.ix_to_tag = saved_model['ix_to_tag']
		self.model.eval()
		self.rnn.eval()
		decoder_output = []
		for sent in tqdm.tqdm(input_data):
			decoder_output.append(self.argmax(sent))
		return decoder_output

if __name__ == '__main__':
	optparser = optparse.OptionParser()
	optparser.add_option("-i", "--inputfile", dest="inputfile", default=os.path.join('data', 'input', 'dev.txt'), help="produce chunking output for this input file")
	optparser.add_option("-t", "--trainfile", dest="trainfile", default=os.path.join('data', 'train.txt.gz'), help="training data for chunker")
	optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join('data', 'chunker'), help="filename without suffix for model files")
	optparser.add_option("-s", "--modelsuffix", dest="modelsuffix", default='.tar', help="filename suffix for model files")
	optparser.add_option("-e", "--epochs", dest="epochs", default=5, help="number of epochs [fix at 5]")
	optparser.add_option("-u", "--unknowntoken", dest="unk", default='[UNK]', help="unknown word token")
	optparser.add_option("-f", "--force", dest="force", action="store_true", default=False, help="force training phase (warning: can be slow)")
	optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
	(opts, _) = optparser.parse_args()

	if opts.logfile is not None:
		logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

	modelfile = opts.modelfile
	if opts.modelfile[-4:] == '.tar':
		modelfile = opts.modelfile[:-4]

	rnnfile = os.path.join('data', 'rnn')
	chunker = LSTMTagger(opts.trainfile, modelfile, rnnfile, opts.modelsuffix, opts.unk)
	# use the model file if available and opts.force is False
	if os.path.isfile(opts.modelfile + opts.modelsuffix) and not opts.force:
		decoder_output = chunker.decode(opts.inputfile)
	else:
		print("Warning: could not find modelfile {}. Starting training.".format(modelfile + opts.modelsuffix), file=sys.stderr)
		chunker.train()
		decoder_output = chunker.decode(opts.inputfile)

	print("\n\n".join([ "\n".join(output) for output in decoder_output ]))
