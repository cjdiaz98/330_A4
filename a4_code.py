import re
import numpy as np
from collections import defaultdict

# load up all of the 19997 documents in the corpus
corpus = sc.textFile ("s3://chrisjermainebucket/comp330_A6/20_news_same_line.txt")

# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x : 'id' in x)

# now we transform it into a bunch of (docID, text) pairs
keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))

# gets list of all documents
allDocs = validLines.map(lambda x : x[x.index('id="') + 4 : x.index('" url=')]) 

# to [("word", (doc, count)),...]
allWordDocOccurrences = None

num_docs = -1

allDictWords = None

allDictWordsSet = None

topWords = None

sortedWordDocOccurrences = None

IDF = None # we'll assign this later

titles_of_interest = ["20 newsgroups/sci.med/58763","20 newsgroups/talk.politics.mideast/75944", "20_newsgroups/comp.graphics/37261"]



def lab5():
	global allDictWords, topWords, allDictWordsSet
	regex = re.compile('[^a-zA-Z]')
	keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))
	allCounts = allWords.reduceByKey (lambda a, b: a + b)
	topWords = allCounts.top (20000, lambda x : x[1])
	twentyK = sc.parallelize(range(20000))
	dictionary = twentyK.map (lambda num: (topWords[num][0] , num))
	allDictWords = dictionary.map(lambda x: x[0])
	allDictWordsSet = set(allDictWords.collect())
	dictionary.top(10)
	return dictionary

def task1():
	global corpus, validLines, keyAndText, dictionary, allDocs, num_docs, numWordsInDoc,allWordDocOccurrences, topWords, allDictWords,titles, allDictWordsSet
	regex = re.compile('[^a-zA-Z]')
	keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	numWordsInDoc = keyAndListOfWords.map(lambda x: (x[0], len(x[1]) ))
	keyAndListOfWords = keyAndListOfWords.map(lambda x: (x[0], list(set(x[1]).intersection(allDictWordsSet)) ))
	num_docs = keyAndListOfWords.count()
	numWordsInDocResult = numWordsInDoc.collect()
	null_word_counts = allDictWords.flatMap(lambda x: (((x,doc), 0) for doc in numWordsInDocResult))
	allWordDocOccurrences = keyAndListOfWords.flatMap(lambda x: ( ((j,x[0]), 1) for j in x[1]))
	allWordDocOccurrences = allWordDocOccurrences.union(null_word_counts)
	allWordDocOccurrences = allWordDocOccurrences.reduceByKey(lambda x,y: x + y)
	newWordDocOccurrences = allWordDocOccurrences.map(lambda x: (x[0][0], (x[0][1],x[1]) ) )
	wordPosDocPairs = sc.parallelize(topWords).join(newWordDocOccurrences)
	docCountTotCountGroupedByDoc = wordPosDocPairs.map(lambda x: x[1]).groupBy(lambda x: x[1][0]) 
	docCountTotCountGroupedByDoc = wordPosDocPairs.map(lambda x: sorted(x[0], True))
	docAndFrequencies = docCountTotCountGroupedByDoc.map(lambda x: (x[0], map(x[1], lambda tup: tup[1][1]) ))
	results = docAndFrequencies.filter(lambda x: x[0] in titles).collect()
	print(results[0])
	print(results[1])
	print(results[2])
	return docAndFrequencies
	# we changed much of this from original!!

def task2():
	global num_docs, docAndFrequencies, numWordsInDoc,allWordDocOccurrences, dictionary, sortedWordDocOccurrences, topWords, IDF, titles_of_interest
	pre_TF_d = docAndFrequencies.join(numWordsInDoc)
	TF_d = pre_TF_d.map(lambda x: (x[0], np.array(x[1][0] / (x[1][1] * 1.))) )
	wordDocSingleOccurrences = allWordDocOccurrences.map(lambda x: (x[0][0], (x[1] > 1) * 1 ))
	wordDocOccurences = wordDocSingleOccurrences.reduceByKey(lambda x,y: x + y)
	sortedWordDocOccurrences = wordDocOccurences.join(topWords)
	sortedWordDocOccurrences = sortedWordDocOccurrences.map(
		lambda x: (x[1][1],x[1][0]) ).sortByKey(ascending = False)
	sortedWordDocOccurrences = sortedWordDocOccurrences.map(lambda x: x[1][1])
	IDF = sortedWordDocOccurrences.map(lambda x: math.log(x / (num_docs * 1.)))
	TF_IDF = TF_d.map(lambda x: (x[0], x[1] * IDF))
	results = TF_IDF.filter(lambda x: x[0] in titles).collect()
	print(results[0])
	print(results[1])
	print(results[2])
	return TF_IDF


def task3(in_string, k):
	global dictionary, allDictWords, topWords, IDF, TF_IDF
	rgx = '[a-zA-Z\']+'
	sampleWords = sc.parallelize(re.findall(rgx, in_string))
	numWordsInSample = sampleWords.count()
	sampleWords = sampleWords.intersection(allDictWords)
	sampleWords = sampleWords.map(lambda x: (x, 1))
	sampleWords = sampleWords.union(allDictWords.map(lambda x: (x, 0)))
	sampleWords = sampleWords.reduceByKey(lambda x,y:x+y)
	sampleWords = topWords.join(sampleWords)
	curr_freq = curr_freq.map(lambda x: x[1])
	curr_freq = curr_freq.sortByKey(ascending = False)
	curr_freq = curr_freq.map(lambda x: x[1])	
	this_tf = np.array(curr_freq) / (numWordsInSample * 1.)
	this_tf_idf = this_tf * IDF
	l2_norm = TF_IDF.map(lambda x: (x[0], calc_l2_norm(np.array(x[1]),this_tf_idf)) )
	rev_l2_norm = l2_norm.map(lambda x: (x[1], x[0]))
	topk = rev_l2_norm.top(k).collect()
	cat_count = defaultdict(int)
	max_score = -1
	for pair in topk:
		cat = get_cat(pair[1])
		cat_count[cat] += 1
		if max_score < cat_count[cat]:
			max_score = cat_count[cat]
	all_max_cat = []
	for key in map_scores:
		if map_scores[key] == max_score:
			all_max_cat.append(key)
	pick_cat = None
	if len(all_max_cat) > 1:
		for i in topk:
			if i[1] in all_max_cat:
				pick_cat = i[1]
				break
	elif len(all_max_cat) == 1:
		pick_cat = all_max_cat[0]
	else:
		print("No max categories")
		return None
	print("Final category: %s" % pick_cat)
	return pick_cat 


def calc_l2_norm(arr1, arr2):
	diff =  abs(arr1 - arr2)**2
	return math.pow(diff, .5)

def get_cat(title):
	category = title.split("/")
	if len(category) > 1:
		return category[1]
	return None

#######	NOTE: I CHANGED THE CODE ABOVE TO LOOK DIFFERENT BECAUSE PYSPARK WOULDN'T ACCEPT IT OTHERWISE

dictionary = lab5()

# [(doc, [frequency vector]),...]
docAndFrequencies = task1()

# [(doc,TF_IDF(i) vector), ...]
TF_IDF = task2()
# ATTENTION!!! IF YOU RUN INTO ANY ERRORS, CHECK THE TODO SECTIONS FIRST!!


task3()




# def lab5():
# 	global allDictWords, topWords
# 	# now we split the text in each (docID, text) pair into a list of words
# 	# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
# 	# we have a bit of fancy regular expression stuff here to make sure that we do not
# 	# die on some of the documents
# 	regex = re.compile('[^a-zA-Z]')
# 	keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# 	# now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# 	# to ("word1", 1) ("word2", 1)...
# 	allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))

# 	# now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
# 	allCounts = allWords.reduceByKey (lambda a, b: a + b)

# 	# and get the top 20,000 words in a local array
# 	# each entry is a ("word1", count) pair
# 	topWords = allCounts.top (20000, lambda x : x[1])

# 	# and we'll create a RDD that has a bunch of (word, dictNum) pairs
# 	# start by creating an RDD that has the number 0 thru 20000
# 	# 20000 is the number of words that will be in our dictionary
# 	twentyK = sc.parallelize(range(20000))

# 	# now, we transform (0), (1), (2), ... to ("mostcommonword", 1) ("nextmostcommon", 2), ...
# 	# the number will be the spot in the dictionary used to tell us where the word is located
# 	# HINT: make use of topWords in the lambda that you supply
# 	dictionary = twentyK.map (lambda num: (topWords[num][0] , num))
# 	allDictWords = dictionary.map(lambda x: x[0])
# 	# finally, print out some of the dictionary, just for debugging
# 	dictionary.top(10)
# 	return dictionary


# def task1():
# 	global corpus, validLines, keyAndText, dictionary, allDocs, num_docs, numWordsInDoc, 
# 	allWordDocOccurrences, topWords, allDictWords
# 		# now we split the text in each (docID, text) pair into a list of words
# 	# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
# 	# we have a bit of fancy regular expression stuff here to make sure that we do not
# 	# die on some of the documents
# 	regex = re.compile('[^a-zA-Z]')
# 	# each entry looks like: (docID, ["word1", "word2", "word3", ...])
# 	keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	
# 	numWordsInDoc = keyAndListOfWords.map(lambda x: (x[0], len(x[1]) )) 
# 	# we use this ^ result for task 2

# 	keyAndListOfWords = keyAndListOfWords.map(lambda x: (x[0], allDictWords.intersection(x[1])) ) 
# 	# get rid of any words not in the dictionary
# 	# TODO: can we intersect RDD with non-RDD? -- check this!!

# 	num_docs = keyAndListOfWords.count()

# 	null_word_counts = allDictWords.flatMap(lambda x: ((x,doc), 0) for doc in allDocs)
	
# 	# first change (docID, ["word1", "word2", "word3", ...]) to [(("word", doc), 1)]
# 	allWordDocOccurrences = keyAndListOfWords.flatMap(lambda x: ((j,x[0]), 1) for j in x[1])


# 	# to (("word",doc), count)
# 	allWordDocOccurrences = allWordDocOccurrences.union(null_word_counts)
# 	allWordDocOccurrences = allWordDocOccurrences.reduceByKey(lambda x,y: x + y)

# 	# to ("word", (doc,count))
# 	newWordDocOccurrences = allWordDocOccurrences.map(lambda x: (x[0][0], (x[0][1],x[1]) ) )

# 	# this gets us the form ( "word", (totalCount1,(doc, count1)) )
# 	wordPosDocPairs = topWords.join(newWordDocOccurrences)
	
# 	# [(doc, [(totalCount1, (doc, count1)), ... ]), ...]
# 	docCountTotCountGroupedByDoc = docCountTotCount.groupBy(lambda x: x[1][1][0])

# 	# [(doc, [(totalCount1, (doc, count1)), ... ]), ...] --> sorted
# 	docCountTotCountGroupedByDoc = wordPosDocPairs.map(lambda x: sorted(x[0], True)) 
# 	# sort in reverse


# 	# [(doc, [ count1, ... ]), ...] --> interior list is sorted
# 	docAndFrequencies = docCountTotCountGroupedByDoc
# 		.map(lambda x: (x[0], map(x[1], lambda tup: tup[1][1]) )) # this gets the interior list sorted
	
# 	results = docAndFrequencies.filter(lambda x: x[0] in titles).collect()
# 	print(results[0])
# 	print(results[1])
# 	print(results[2])
	
# 	return docAndFrequencies


# def task2():
# 	global num_docs, docAndFrequencies, numWordsInDoc,
# 	 allWordDocOccurrences, dictionary, sortedWordDocOccurrences, topWords, IDF, titles_of_interest

# 	# array [(doc,([count1,...],numWordsTotal)), ...]
# 	pre_TF_d = docAndFrequencies.join(numWordsInDoc)

# 	# TF vector
# 	TF_d = pre_TF_d.map(lambda x: (x[0], np.array(x[1][0] / (x[1][1] * 1.))) )


# 	# trim all counts to 0 or 1. 
# 	#gets us an rdd of form ("word", 1 or 0)
# 		# --> corresponding to whether or not a word has been in a document
# 	wordDocSingleOccurrences = allWordDocOccurrences.map(lambda x: (x[0][0], (x[1] > 1) * 1 ))
	
# 	# [(word1, occurrences),...]
# 	wordDocOccurences = wordDocSingleOccurrences.reduceByKey(lambda x,y: x + y)

# 	#gets us an rdd of form [("word", (totalDoc, totalCount),...]
# 	sortedWordDocOccurrences = wordDocOccurences.join(topWords)

# 	#gets us an rdd of form [(totalCount,totalDoc), ...] sorted
# 	sortedWordDocOccurrences = sortedWordDocOccurrences.map(
# 		lambda x: (x[1][1],x[1][0]) ).sortByKey(ascending = False)

# 	#gets us an rdd of form [totalWordDoc1, ...]
# 	sortedWordDocOccurrences = sortedWordDocOccurrences.map(lambda x: x[1][1])

# 	#gets us an rdd of form [IDF(1), IDF(2) ...]
# 	IDF = sortedWordDocOccurrences.map(lambda x: math.log(x / (num_docs * 1.)))

# 	# [(doc, TF_IDF vector)]
# 	TF_IDF = TF_d.map(lambda x: (x[0], x[1] * IDF))

# 	results = TF_IDF.filter(lambda x: x[0] in titles).collect()
# 	print(results[0])
# 	print(results[1])
# 	print(results[2])
# 	return TF_IDF


# def task3(in_string, k):
# 	""""""
# 	global dictionary, allDictWords, topWords, IDF, TF_IDF
# 	# need list [category of doc 1,...]
# 	# need list [cat1 frequency, ...]

# 	# possibly create function that will take in text 
# 	# and construct a list of dictionary word frequencies
# 	rgx = '[a-zA-Z\']+'
# 	sampleWords = sc.parallelize(re.findall(rgx, in_string))

# 	numWordsInSample = sampleWords.count()

# 	sampleWords = sampleWords.intersection(allDictWords)

# 	sampleWords = sampleWords.map(lambda x: (x, 1))


# 	# [(word, 0 or 1),...]
# 	sampleWords = sampleWords.union(allDictWords.map(lambda x: (x, 0)))
# 	# [(word, totSampleCount),...]
# 	sampleWords = sampleWords.reduceByKey(lambda x,y:x+y)

# 	# should get you [(word,(total, thisCount)),...]
# 	sampleWords = topWords.join(sampleWords)

# 	# gets you [(total, count),...]
# 	curr_freq = curr_freq.map(lambda x: x[1])

# 	# gets you [(total, count),...] sorted
# 	curr_freq = curr_freq.sortByKey(ascending = False)
	
# 	# gets you [count,...]
# 	curr_freq = curr_freq.map(lambda x: x[1])	
	
# 	this_tf = np.array(curr_freq) / (numWordsInSample * 1.)

# 	this_tf_idf = this_tf * IDF

# 	l2_norm = TF_IDF.map(lambda x: (x[0], calc_l2_norm(np.array(x[1]),this_tf_idf)) )

# 	# organize by norm so we can pick top ones
# 	rev_l2_norm = l2_norm.map(lambda x: (x[1], x[0]))

# 	# [(score, doc),...]
# 	topk = rev_l2_norm.top(k).collect()

# 	cat_count = defaultdict(int)
# 	max_score = -1

# 	for pair in topk:
# 		cat = get_cat(pair[1])
# 		cat_count[cat] += 1
# 		if max_score < cat_count[cat]:
# 			max_score = cat_count[cat]

# 	all_max_cat = []
# 	for key in map_scores:
# 		if map_scores[key] == max_score:
# 			all_max_cat.append(key)

# 	pick_cat = None
# 	if len(all_max_cat) > 1:
# 		# tie breaking
# 		for i in topk:
# 			if i[1] in all_max_cat:
# 				pick_cat = i[1]
# 				break
# 				# pick the category who has the largest score (between those tied)

# 	elif len(all_max_cat) == 1:
# 		pick_cat = all_max_cat[0]
# 	else:
# 		print("No max categories")
# 		return None

# 	print("Final category: %s" % pick_cat)
# 	return pick_cat 