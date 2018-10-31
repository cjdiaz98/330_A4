import re
import numpy as np
from collections import defaultdict
from heapq import heapify, heappush, heappop

# load up all of the 19997 documents in the corpus
corpus = sc.textFile ("s3://chrisjermainebucket/comp330_A6/20_news_same_line.txt")

# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x : 'id' in x)

# now we transform it into a bunch of (docID, text) pairs
keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))

# gets list of all documents
allDocs = validLines.map(lambda x : x[x.index('id="') + 4 : x.index('" url=')]) 

num_docs = -1

def lab5():
	# now we split the text in each (docID, text) pair into a list of words
	# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
	# we have a bit of fancy regular expression stuff here to make sure that we do not
	# die on some of the documents
	regex = re.compile('[^a-zA-Z]')
	keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

	# now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
	# to ("word1", 1) ("word2", 1)...
	allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))

	# now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
	allCounts = allWords.reduceByKey (lambda a, b: a + b)

	# and get the top 20,000 words in a local array
	# each entry is a ("word1", count) pair
	topWords = allCounts.top (20000, lambda x : x[1])

	# and we'll create a RDD that has a bunch of (word, dictNum) pairs
	# start by creating an RDD that has the number 0 thru 20000
	# 20000 is the number of words that will be in our dictionary
	twentyK = sc.parallelize(range(20000))

	# now, we transform (0), (1), (2), ... to ("mostcommonword", 1) ("nextmostcommon", 2), ...
	# the number will be the spot in the dictionary used to tell us where the word is located
	# HINT: make use of topWords in the lambda that you supply
	dictionary = twentyK.map (lambda num: (topWords[num][0] , num))

	# finally, print out some of the dictionary, just for debugging
	dictionary.top (10)
	return dictionary

dictionary = lab5()

def task1():
	global corpus, validLines, keyAndText, dictionary, allDocs, num_docs
	keyAndText.map
		# now we split the text in each (docID, text) pair into a list of words
	# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
	# we have a bit of fancy regular expression stuff here to make sure that we do not
	# die on some of the documents
	regex = re.compile('[^a-zA-Z]')
	keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	num_docs = len(keyAndListOfWords)

	null_word_counts = dictionary.flatMap(lambda x: (x, (doc,0)) for doc in allDocs)
	
	# now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
	# to ("word", (doc, 1))
	allWordDocOccurrences = keyAndListOfWords.flatMap(lambda x: (j, (x[0], 1)) for j in x[1])

	# to ("word", (doc, count))
	allWordDocOccurrences = allWordDocOccurrences.union(null_word_counts)

	# this gets us the form ("word", (totalCount, (doc, count) ))
	wordPosDocPairs = dictionary.join(allWordDocOccurrences)
	wordPosDocPairs = wordPosDocPairs.map(lambda x: ( (x[0], x[1][1][0]), (x[1][1], x[1][1][1] )))
	# (("word",doc), (pos,count))

	# (("word",doc), (pos,totalCount))
	wordPosDocTotalPairs = wordPosDocPairs.reduceByKey(lambda x,y: (x[1][0], x[1][1] + y[1][1] ) )

	# (doc, (pos,totalCount))
	wordDocTotal = wordPosDocTotalPairs.map(lambda x: (x[0][1], x[1]))

	# (doc, [(pos1, totalCount1),...]) 

	# (doc, [totalCount1,...])
	# TODO: do we want to represent this as a numpy array instead?




	# # now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
	# # to (("word",doc), (pos, 1))
	# allWordDocOccurrences = keyAndListOfWords.flatMap(lambda x: ((j, x[0]), 1) for j in x[1])

	# # get form
	# null_word_counts = dictionary.flatMap(lambda x: ((x[0],doc), (x[1], 0)) for doc in allDocs)

	# wordPosDocPairs = dictionary.rightOuterJoin(allWordDocOccurrences)
	# # ("word",(pos, (doc,1) ) )
	# wordPosDocPairs = wordPosDocPairs.union(null_word_counts)
	# # add all null word counts

	# wordPosDocTotalPairs = wordPosDocPairs.reduceByKey(lambda x,y: (x[0], (x[1][0], x[1][1] + y[1][1]) ) )


def task2():
	global num_docs
	# For calculating TF(i, d)
	# (doc, (pos,totalCount))
	# (doc, numWords)


	# get IDF vector
	# Need a d by i matrix consisting of counts per document

def insert_heap(k, score_heap, elem):
	""""""
	# (see following link) https://www.geeksforgeeks.org/heap-queue-or-heapq-in-python/
	heappush(li,elem)
	if len(score_heap) > k:
		heappop(li)


def task3(in_string, k):
	""""""

	# need list [category of doc 1,...]
	# need list [cat1 frequency, ...]

	# possibly create function that will take in text 
	# and construct a list of dictionary word frequencies


	# calculate the word frequencies of the current text string
	curr_freq = 

	# create a heap of elements of form (score, cat_id) -- ACTUALLY PROBABLY NOT GONNA DO THIS
	# score_heap = [] 

	# insert into the heap 
	# for :
	# 	# iterate through categories
	# 	doc_freq = 
	# 	insert_pair = (l2_norm(doc_freq, curr_freq), ) # (l2norm, cat)
	# 	insert_heap(k, score_heap, insert_pair)


	# sort the heap by decreasing score
	sort(score_heap)

	# check for any ties in the heap
	map_scores = defaultdict(int)
	max_score = -1

	for pair in score_heap:
		map_scores[pair[1]] += 1
		if max_score < pair[0]:
			max_score = pair[0]

	all_max_cat = []
	for key in map_scores:
		if map_scores[key] == max_score:
			all_max_cat.append(key)

	pick_cat = None
	if len(all_max_cat) > 1:
		# tie breaking
		for i in score_heap:
			if i[1] in all_max_cat:
				pick_cat = i[1]
				break
				# pick the category who has the largest score (between those tied)

	elif len(all_max_cat) == 1:
		pick_cat = all_max_cat[0]
	else:
		print("No max categories")



def l2_norm(countsA, countsB):
	""""""
	differences = countsA - countsB
	differences = differences ** 2 # square the differences

	diff_sum = np.sum(differences)
	return math.pow(diff_sum, .5)