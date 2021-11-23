import re
from rulebased_tokenizer import rulebased_tokenizer
from rulebased_sentencesplitter import rulebased_sentencesplitter
from sklearn.linear_model import LogisticRegression
space_regexp = re.compile(r'^\s+')
PUNCT_SEQ_RE = re.compile(r'[-!\'#%&`()\[\]*+,.\\/:;<=>?@^$_{|}~]+')

"""
def window_slide(words):
    data=[]
    for i, window in enumerate(words):
        if i==len(words)-1:
            data.append([words[i],'PAD'])
        else:
            data.append([words[i], words[i + 1]])

    return data
"""
bb=rulebased_sentencesplitter()
cc= rulebased_tokenizer()

def is_next_digit(word):
    if word.isdigit():
        return 1
    return 0

def next_isupper(word):
    if word.isupper():
        return 1
    return 0

def next_islower(word):
    if word.islower():
        return 1
    return 0

def is_next_char(word):
    if len(word)>0:
        return 1
    return 0

def is_next_space(word):
    if space_regexp.fullmatch(word):
        return 1
    return 0

def is_next_punct(word):
    if PUNCT_SEQ_RE.fullmatch(word):
        return 1
    return 0

def is_next_capitalupper(word):
    if word[0].isupper():
        return 1
    return 0

def feature_set_1(words, X):


    for i,w in enumerate(words):
        token_feat=[]
        #print(words[i])
        if i < len(words)-1:
                token_feat.append(is_next_digit(words[i+1]))
                token_feat.append(next_isupper(words[i+1]))
                token_feat.append(next_islower(words[i+1]))
                token_feat.append(is_next_space(words[i+1]))
                token_feat.append(is_next_punct(words[i+1]))
                token_feat.append(is_next_capitalupper(words[i+1]))
                token_feat.append(is_next_char(words[i+1]))
                #token_feat.append(cc.abbreviation(words[i+1]))
        else:
            token_feat.append(0)
            token_feat.append(0)
            token_feat.append(0)
            token_feat.append(0)
            token_feat.append(0)
            token_feat.append(0)
            token_feat.append(0)
            #token_feat.append(0)
        X.append(token_feat)
    return X


def feature_set_2(words, X):


    for i,w in enumerate(words):
        token_feat=[]
        #print(words[i])
        if i < len(words)-1:
                #token_feat.append(is_next_digit(words[i+1]))
                #token_feat.append(next_isupper(words[i+1]))
                #token_feat.append(next_islower(words[i+1]))
                #token_feat.append(is_next_space(words[i+1]))
                #token_feat.append(is_next_punct(words[i+1]))
                token_feat.append(is_next_capitalupper(words[i+1]))
                #token_feat.append(is_next_char(words[i+1]))
                token_feat.append(cc.abbreviation(words[i]))
        else:
            #token_feat.append(0)
            #token_feat.append(0)
            #token_feat.append(0)
            #token_feat.append(0)
            #token_feat.append(0)
            token_feat.append(0)
            #token_feat.append(0)
            token_feat.append(cc.abbreviation(words[i]))
        X.append(token_feat)
    return X


def read_traindata(filename):
    y = []
    X= []
    with open(filename,"r", encoding="utf-8") as file:
        for line in file:
            if "# text = " in line:
                #print(line.rstrip())
                #line=line.encode('utf-8')

                doc = line.rstrip().replace("# text = ","")
                sentences=bb.sent_tokenize(doc, bb.turkish_abbreviation_set)
                for sent in sentences:
                    tokens = cc._tokenize(sent)
                    words=[]
                    for token in tokens:
                        words.append(token)
                        y.append("N")
                    y[-1]="L"
                    feature_set_2(words, X)
    return X,y

def testdata(filename):
    y = []
    y_test=[]
    X= []
    with open(filename,"r", encoding="utf-8") as file:
        for line in file:
            if "# text = " in line:
                #print(line.rstrip())
                #line=line.encode('utf-8')

                doc = line.rstrip().replace("# text = ","")
                sentences=bb.sent_tokenize(doc, bb.turkish_abbreviation_set)
                for sent in sentences:
                    tokens = cc._tokenize(sent)
                    words=[]
                    for token in tokens:
                        words.append(token)
                        y.append("N")
                    y[-1]="L"
                    feature_set_2(words, X)

            else:

                y_test.append(line.rstrip())
    return X,y,y_test

X,y=read_traindata("tr_penn-ud-train.conllu")
print(len(X))
print(len(X[0]))
print(len(y))
#print(X)
clf = LogisticRegression(random_state=0).fit(X, y)

X_test,y_test,y_test_gold=testdata("chosen_test.txt")
y_pred=clf.predict(X_test)

print(len(X_test))
print(len(y_pred))

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print(confusion_matrix(y_test_gold, y_pred, labels=["N", "L"]))



# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test_gold, y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
#precision = precision_score(y_test, y_pred)
#print('Precision: %f' % precision)
# recall: tp / (tp + fn)
#recall = recall_score(y_test, y_pred)
#print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
#f1 = f1_score(y_test, y_pred)
#print('F1 score: %f' % f1)