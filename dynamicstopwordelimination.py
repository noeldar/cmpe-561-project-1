from collections import Counter
import os
import string
import numpy as np
from rulebased_tokenizer import *
import copy
import pandas as pd
import pickle
import re
import math
import builtins
from forbiddenfruit import curse
from TurkishStemmer import TurkishStemmer
stemmer = TurkishStemmer()
from rulebased_tokenizer import *

lcase_table = tuple(u'abcçdefgğhıijklmnoöprsştuüvyz')
ucase_table = tuple(u'ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ')

def upper(data):
    data = data.replace('i',u'İ')
    data = data.replace(u'ı',u'I')
    result = ''
    for char in data:
        try:
            char_index = lcase_table.index(char)
            ucase_char = ucase_table[char_index]
        except:
            ucase_char = char
        result += ucase_char
    return result

def lower(data):
    data = data.replace(u'İ',u'i')
    data = data.replace(u'I',u'ı')
    result = ''
    for char in data:
        try:
            char_index = ucase_table.index(char)
            lcase_char = lcase_table[char_index]
        except:
            lcase_char = char
        result += lcase_char
    return result

def capitalize(data):
    return data[0].upper() + data[1:].lower()

def title(data):
    return " ".join(map(lambda x: x.capitalize(), data.split()))

def convert_lower_case(data):
    return lower(data)

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def tokenize(sentence):
    rt = rulebased_tokenizer()
    tokens = rt._tokenize(sentence)
    tmp = []
    for token in tokens:
        tmp.append(token)
    return tmp

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    tokens = tokenize(str(data))
    for i in range(len(tokens)):
        tokens[i]=stemmer.stem(tokens[i])
    return ' '.join(tokens)

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


def computeIDF(documents):
    import math
    N = len(documents)

    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf


sentences_train=[]


with open("tr_penn-ud-train.conllu","r", encoding="utf-8") as file:
    for line in file:
        if "# text = " in line:
            sentences_train.append(line.strip().replace("# text = ",""))


sentences_test=[]


with open("tr_penn-ud-test.conllu","r", encoding="utf-8") as file:
    for line in file:
        if "# text = " in line:
            sentences_test.append(line.strip().replace("# text = ",""))


documentA = ""

for sent in sentences_train:

    documentA = documentA  +" "+str(preprocess(sent))
    #print(str(preprocess(sent)))

documentB = ""

for sent in sentences_test:

    documentB  = documentB  +" "+str(preprocess(sent))




bagOfWordsA = documentA.split(' ')
bagOfWordsB = documentB.split(' ')
uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))

numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
    numOfWordsA[word] += 1
numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
    numOfWordsB[word] += 1

tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)
idfs = computeIDF([numOfWordsA, numOfWordsB])
tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)
df = pd.DataFrame([tfidfA, tfidfB])
#print(df.mean(axis = 0))
weights = np.asarray(df.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': df.columns.values, 'weight': weights})
weights_df.sort_values(by='weight', ascending=False).to_csv('tfidf_cud_values.txt', index=False)
terms=weights_df["term"].values
stopwords=terms[-3000:]
#stopwords=terms[:2000]
#print(stopwords)

"""file1 = open('stop_words_turkish.txt', 'r', encoding="utf-8")
Lines = file1.readlines()

count = 0
# Strips the newline character
stopwordss=[]
for line in Lines:
    #count += 1
    #print("Line{}: {}".format(count, line.strip()))
    stopwordss.append(line.strip())

stopwords_set = list(set(stopwordss))

cnt=0
for s in stopwords:
    if s in stopwords_set:
        cnt=cnt+1

print(float(cnt)/3000.0)"""