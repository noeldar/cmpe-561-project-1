import re
from naivebayes import *
space_regexp = re.compile(r'^\s+')
PUNCT_SEQ_RE = re.compile(r'[-!\'#%&`()\[\]*+,.\\/:;<=>?@^$_{|}~]+')

def is_next_digit(sentence, index):
    if sentence[index:index+1].isdigit():
        return 1
    return 0

def next_isupper(sentence, index):
    if sentence[index:index+1].isupper():
        return 1
    return 0

def next_islower(sentence, index):
    if sentence[index:index+1].islower():
        return 1
    return 0

def is_next_space(sentence, index):
    if space_regexp.fullmatch(sentence[index:index+1]):
        return 1
    return 0

def is_next_punct(sentence, index):
    if PUNCT_SEQ_RE.fullmatch(sentence[index:index+1]):
        return 1
    return 0


sentences_train=[]
tokens_train = []
tokens = []
with open("tr_penn-ud-train.conllu","r", encoding="utf-8") as file:
    for line in file:
        if "# text = " in line:
            sentences_train.append(line.strip().replace("# text = ",""))
            tokens_train.append(tokens)
            tokens = []
        elif line[0].isdigit() and "\t" in line:
            tts=line.strip().split("\t")
            tokens.append(tts[1])
tokens_train.append(tokens)
tokens = []
tokens_train = tokens_train[1:]
#print(tokens_train)

X_train=[]
Y_train=[]
inx=0
for sentence in sentences_train:

    y=[]
    for match in re.finditer(PUNCT_SEQ_RE, sentence):

        x=[]

        if match.end() < len(sentence):
            x.append(is_next_digit(sentence, match.end()))
            x.append(next_isupper(sentence, match.end()))
            x.append(next_islower(sentence, match.end()))
            #x.append(is_next_space(sentence, match.end()))
            #x.append(is_next_punct(sentence, match.end()))
        else:
            x.append(0)
            x.append(0)
            x.append(0)
            #x.append(0)
            #x.append(0)

        if sentence[match.start():match.end()] in  tokens_train[inx]:
            Y_train.append(1)
        else:
            Y_train.append(0)

        X_train.append(x)
    #Y_train.append(y)
    inx = inx + 1


#######################################################################################################################

sentences_test=[]
tokens_test = []
tokens = []
with open("tr_penn-ud-test.conllu","r", encoding="utf-8") as file:
    for line in file:
        if "# text = " in line:
            sentences_test.append(line.strip().replace("# text = ",""))
            tokens_test.append(tokens)
            tokens = []
        elif line[0].isdigit() and "\t" in line:
            tts=line.strip().split("\t")
            tokens.append(tts[1])

tokens_test.append(tokens)
tokens = []
tokens_test = tokens_test[1:]
#print(tokens_test)

X_test=[]
Y_test=[]
inx=0
for sentence in sentences_test:

    y=[]
    for match in re.finditer(PUNCT_SEQ_RE, sentence):
        x=[]
        if match.end() < len(sentence):
            x.append(is_next_digit(sentence, match.end()))
            x.append(next_isupper(sentence, match.end()))
            x.append(next_islower(sentence, match.end()))
            #x.append(is_next_space(sentence, match.end()))
            #x.append(is_next_punct(sentence, match.end()))
        else:
            x.append(0)
            x.append(0)
            x.append(0)
            #x.append(0)
            #x.append(0)

        if sentence[match.start():match.end()] in  tokens_test[inx]:
            Y_test.append(1)
        else:
            Y_test.append(0)

        X_test.append(x)
    #Y_test.append(y)
    inx = inx + 1


print(len(X_test))
print(len(Y_test))
print(len(X_train))
print(len(Y_train))

if 1 in Y_test:
    print("lolo")


tr = pd.DataFrame(X_train, columns=["1","2","3"])
test = pd.DataFrame(X_test, columns=["1","2","3"])
nb=naivebayes()
pred= nb.predict(test, tr, Y_train)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print(confusion_matrix(Y_test, pred))

accuracy = accuracy_score(Y_test, pred)
precision = precision_score(Y_test, pred)
recall = recall_score(Y_test, pred)
f1 = f1_score(Y_test, pred)
print('Accuracy: %f' % accuracy)
print('precision: %f' % precision)
print('recall: %f' % recall)
print('f1: %f' % f1)