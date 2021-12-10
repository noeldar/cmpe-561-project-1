# Using readlines()
file1 = open('stop_words_turkish.txt', 'r', encoding="utf-8")
Lines = file1.readlines()

count = 0
# Strips the newline character
stopwords=[]
for line in Lines:
    #count += 1
    #print("Line{}: {}".format(count, line.strip()))
    stopwords.append(line.strip())

stopwords_set = list(set(stopwords))
print(stopwords_set)