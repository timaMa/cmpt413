import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10

class Entry:
	def __init__(self, word, startpt, logP, backpointer):
		self.word = word
		self.startpt = startpt
		self.logP = logP
		self.backpointer = backpointer
		
	def __eq__(self, entry):
		if not isinstance(entry, type(self)): return False
		return (self.word == entry.word) and (self.startpt == entry.startpt) and (self.logP == entry.logP)
		
	def __gt__(self, entry):
		return self.logP <= entry.logP
		
	def __lt__(self, entry):
		return self.logP > entry.logP

class Segment:

	def __init__(self, uniPw, biPw):
		self.uniPw = uniPw
		self.biPw = biPw

	def segment(self, text):
		### Initialize the heap
		heap = []
		for i in range(len(text)):
			word = text[0:i+1]
			if word in self.uniPw or len(word) <= 4:
				heapq.heappush(heap, Entry(word, 0, log10(self.uniPw(word)), None))
				
		### Iteratively fill in chart[i] for all i
		chart = {}
		for i in range(len(text)):
			chart[i] = Entry(None, None, None, None)
		
		while len(heap) > 0:
			entry = heapq.heappop(heap)
			endindex = entry.startpt + len(entry.word) - 1
			if chart[endindex].backpointer is not None:
				preventry = chart[endindex].backpointer
				if entry.logP > preventry.logP:
					chart[endindex] = entry
				if entry.logP <= preventry.logP:
					continue
			else:
				chart[endindex] = entry
				
			for i in range(endindex + 1, len(text)):
				word = text[endindex + 1 : i + 1]
				if word in self.uniPw or len(word) <= 4:
					wordPair = entry.word + " " + word
					uniP = self.uniPw(word)

					if wordPair in self.biPw and entry.word in self.uniPw:
						biP = self.biPw(wordPair) / self.uniPw(word)
						newentry = Entry(word, endindex + 1, entry.logP + log10(biP), entry)
					else:
						newentry = Entry(word, endindex + 1, entry.logP + log10(uniP), entry)
					if newentry not in heap:
						heapq.heappush(heap, newentry)
			
		### Get the best segmentation
#        print(chart)
#        print(text)
		finalindex = len(chart)
		finalentry = chart[finalindex - 1]
#        print(finalentry)
		segmentation = []
		while finalentry != None:
			segmentation.insert(0, finalentry.word)
			finalentry = finalentry.backpointer
#        print(segmentation)

		return segmentation

#### Support functions (p. 224)

def product(nums):
	"Return the product of a sequence of numbers."
	return reduce(operator.mul, nums, 1)

class Pdist(dict):
	"A probability distribution estimated from counts in datafile."
	def __init__(self, data=[], N=None, missingfn=None):
		for key,count in data:
			self[key] = self.get(key, 0) + int(count)
		self.N = float(N or sum(self.values()))
		self.missingfn = missingfn or (lambda k, N: 1./(N * 9000 ** len(k)))
	def __call__(self, key): 
		if key in self: return self[key]/self.N  
		else: return self.missingfn(key, self.N)

def datafile(name, sep='\t'):
	"Read key,value pairs from file."
	with open(name) as fh:
		for line in fh:
			(key, value) = line.split(sep)
			yield (key, value)


if __name__ == '__main__':
	optparser = optparse.OptionParser()
	optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
	optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
	optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
	optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
	(opts, _) = optparser.parse_args()

	if opts.logfile is not None:
		logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

	uniPw = Pdist(data=datafile(opts.counts1w))
	biPw = Pdist(data=datafile(opts.counts2w)) 
	segmenter = Segment(uniPw, biPw)
	with open(opts.input) as f:
		for line in f:
			print(" ".join(segmenter.segment(line.strip())))
