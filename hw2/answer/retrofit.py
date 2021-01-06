import os, sys, optparse
import tqdm
import pymagnitude
from pymagnitude import converter
import numpy as np
import subprocess


''' Read the wordnet'''
def read_lexicon(filename):
	lexicon = {}
	for line in open(filename, 'r'):
		words = line.lower().strip().split()
		if words[0] not in lexicon:
			lexicon[words[0]] = [word for word in words[1:]]
		else:
			lexicon[words[0]] = lexicon[words[0]] + [word for word in words[1:]]
	return lexicon



def retrofit(wvecs, lexicon, T = 20, alpha = 2.0, beta = 1.0):
	vocab = set()
	newVecs = {}
	for word, vec in wvecs:
		vocab.add(word)
		newVecs[word] = vec

	for i in range(T):
		for index, word in enumerate(vocab):
			tmpVec = np.zeros(newVecs[word].shape)
			if word in lexicon:
				count = 0
				neighbors = lexicon[word]
				for w in neighbors:
					if w in newVecs:
						tmpVec += beta * newVecs[w]
						count += 1

				newVecs[word] = ((tmpVec + (alpha * wvecs.query(word)))) / (count + alpha)

	return newVecs

def create_retrofit(retrofit_exist):
	if os.getcwd().split('/')[-1] == 'answer':
		path = os.path.dirname(os.getcwd())
	else:
		path = os.getcwd()
	
	lexicon_path = os.path.join(path, 'data/lexicons', 'ppdb-xl.txt')
	lexicon = read_lexicon(lexicon_path)

	if not retrofit_exist:
		wvec = pymagnitude.Magnitude(os.path.join(path, 'data', 'glove.6B.100d.magnitude'))
		txt_file = os.path.join(path, 'data', 'glove.6B.100d.retrofit.txt')
		newvecs = retrofit(wvec, lexicon)
		with open(txt_file, 'w') as f:
		    vSize = len(newvecs)
		    for word, emb in newvecs.items():
		        s = word
		        for num in emb:
		            s = s + " " + str(num)
		        s = s + '\n'
		        f.write(s)
		target_file = os.path.join(path, 'data', 'glove.6B.100d.retrofit.magnitude')
		converter.convert(txt_file, target_file)

	wvecs = pymagnitude.Magnitude(os.path.join(path, 'data', 'glove.6B.100d.retrofit.magnitude'))

	return lexicon, wvecs

if __name__ == '__main__':
	create_retrofit()
	print("Successfully create magnitude file")

