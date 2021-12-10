import re
from rulebased_sentencesplitter import rulebased_sentencesplitter
from sklearn.linear_model import LogisticRegression
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
rss = rulebased_sentencesplitter()

with open("tr_penn-ud-train.conllu","r", encoding="utf-8") as file:
    for line in file:
        if "# text = " in line:
            sentences_train.append(line.strip().replace("# text = ",""))


X_train=[]
Y_train=[]

for sentence in sentences_train:
    sents = rss.sent_tokenize(sentence, rss.turkish_abbreviation_set)
    ends = []

    for sent in sents:
        ii=sentence.find(sent)
        ends.append(ii+len(sent))

    for match in re.finditer(PUNCT_SEQ_RE, sentence):

        x=[]

        if match.end() < len(sentence):
            x.append(is_next_digit(sentence, match.end()))
            x.append(next_isupper(sentence, match.end()))
            #x.append(next_islower(sentence, match.end()))
            #x.append(is_next_space(sentence, match.end()))
            x.append(is_next_punct(sentence, match.end()))
        else:
            x.append(0)
            x.append(0)
            #x.append(0)
            #x.append(0)
            x.append(0)

        if match.end() in  ends:
            Y_train.append(1)
        else:
            Y_train.append(0)

        X_train.append(x)

#######################################################################################################################

sentences_test=[]


with open("tr_penn-ud-test.conllu","r", encoding="utf-8") as file:
    for line in file:
        if "# text = " in line:
            sentences_test.append(line.strip().replace("# text = ",""))


X_test=[]
Y_test=[]

for sentence in sentences_test:
    sents = rss.sent_tokenize(sentence, rss.turkish_abbreviation_set)
    ends = []

    for sent in sents:
        ii=sentence.find(sent)
        ends.append(ii+len(sent))

    for match in re.finditer(PUNCT_SEQ_RE, sentence):

        x=[]

        if match.end() < len(sentence):
            x.append(is_next_digit(sentence, match.end()))
            x.append(next_isupper(sentence, match.end()))
            #x.append(next_islower(sentence, match.end()))
            #x.append(is_next_space(sentence, match.end()))
            x.append(is_next_punct(sentence, match.end()))
        else:
            x.append(0)
            x.append(0)
            #x.append(0)
            #x.append(0)
            x.append(0)

        if match.end() in  ends:
            Y_test.append(1)
        else:
            Y_test.append(0)

        X_test.append(x)


clf = LogisticRegression(random_state=0).fit(X_train, Y_train)

y_pred=clf.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print(confusion_matrix(Y_test, y_pred))



# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)
print('Accuracy: %f' % accuracy)
print('precision: %f' % precision)
print('recall: %f' % recall)
print('f1: %f' % f1)
