import unittest
from zemberek import (
    TurkishSpellChecker,
    TurkishSentenceNormalizer,
    TurkishSentenceExtractor,
    TurkishMorphology,
    TurkishTokenizer
)

from rulebased_tokenizer import rulebased_tokenizer
from rulebased_sentencesplitter import rulebased_sentencesplitter
from datasets import load_dataset



class MyTestCase:
    def my_read_data(self,filename):
        bb = rulebased_sentencesplitter()
        cc = rulebased_tokenizer()
        y_all = []
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                if "# text = " in line:
                    # print(line.rstrip())
                    # line=line.encode('utf-8')

                    doc = line.rstrip().replace("# text = ", "")
                    sentences = bb.sent_tokenize(doc, bb.turkish_abbreviation_set)
                    y=[]
                    for sent in sentences:
                        tokens = cc._tokenize(sent)
                        for token in tokens:
                            y.append("N")
                        y[-1] = "L"

                    y_all.append(y)

        return y_all

    def my_read_list(self,liste):
        bb = rulebased_sentencesplitter()
        cc = rulebased_tokenizer()
        y_all = []
        for line in liste:
                if "# text = " not in line:
                    # print(line.rstrip())
                    # line=line.encode('utf-8')

                    doc = line.rstrip().replace("# text = ", "")
                    sentences = bb.sent_tokenize(doc, bb.turkish_abbreviation_set)
                    y=[]
                    for sent in sentences:
                        tokens = cc._tokenize(sent)
                        for token in tokens:
                            y.append("N")
                        y[-1] = "L"

                    y_all.append(y)

        return y_all

    def read_split(self, dataset):
        texts = []
        labels = []
        # values_dataset = list(dataset.values())
        for i in dataset:
            texts.append(i["sentence"])
            labels.append(i["sentiment"])

        return texts[0:500]


    def zemberek_read_data(self, filename):
        bb = TurkishSentenceExtractor()
        cc = TurkishTokenizer.DEFAULT
        y_all = []
        sents=[]
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                if "# text = " in line:
                    # print(line.rstrip())
                    # line=line.encode('utf-8')

                    doc = line.rstrip().replace("# text = ", "")
                    sents.append(doc)
                    sentences = bb.from_paragraph(doc)
                    y = []
                    for sent in sentences:
                        tokens = cc.tokenize(sent)
                        for token in tokens:
                            y.append("N")
                        y[-1] = "L"

                    y_all.append(y)

        return sents, y_all

    def zemberek_list_data(self, liste):
        bb = TurkishSentenceExtractor()
        cc = TurkishTokenizer.DEFAULT
        y_all = []
        sents=[]
        for line in liste:
                if "# text = " not in line:
                    # print(line.rstrip())
                    # line=line.encode('utf-8')

                    doc = line.rstrip().replace("# text = ", "")
                    sents.append(doc)
                    sentences = bb.from_paragraph(doc)
                    y = []
                    for sent in sentences:
                        tokens = cc.tokenize(sent)
                        for token in tokens:
                            y.append("N")
                        y[-1] = "L"

                    y_all.append(y)

        return sents, y_all


if __name__ == '__main__':
    obj=MyTestCase()
    """
    snts, y_zemberek = obj.zemberek_read_data("tr_penn-ud-test.conllu")
    y_my = obj.my_read_data("tr_penn-ud-test.conllu")
    print(len(y_zemberek))
    indexes=[]
    for i in range(len(y_zemberek)):
        if len(y_zemberek[i])==len(y_my[i]):
            indexes.append(i)

    print(len(indexes))
    snts_chosen=[]
    y_zemberek_chosen = []
    for i in indexes:
        snts_chosen.append(snts[i])
        y_zemberek_chosen.append(y_zemberek[i])
    print(len(snts_chosen))
    print(len(y_zemberek_chosen))

    toplam=0
    match=0
    for i in indexes:
        for j in range(len(y_zemberek[i])):
            if y_zemberek[i][j]==y_my[i][j]:
                match=match+1
            toplam=toplam+1

    print("lololoooloololololololo")
    print(match)
    print(toplam)
    print("lololoooloololololololo")



    all_lines=[]
    for i in range(len(snts_chosen)):
        all_lines.append("# text = "+snts_chosen[i])
        for j in range(len(y_zemberek_chosen[i])):
            all_lines.append(y_zemberek_chosen[i][j])

    textfile = open("chosen_test_2.txt", "w", encoding="utf-8")
    for element in all_lines:
        textfile.write(element + "\n")
    textfile.close()
    """

    #####################################################################################################
    dataset = load_dataset('turkish_product_reviews', split="train[:10%]")
    texts = obj.read_split(dataset)

    snts, y_zemberek = obj.zemberek_list_data(texts)
    y_my = obj.my_read_list(texts)
    indexes = []
    for i in range(len(y_zemberek)):
        if len(y_zemberek[i]) == len(y_my[i]):
            indexes.append(i)
        else:
            print(len(y_zemberek[i]))
            print(len(y_my[i]))

    print(len(indexes))
    snts_chosen = []
    y_zemberek_chosen = []
    for i in indexes:
        snts_chosen.append(snts[i])
        y_zemberek_chosen.append(y_zemberek[i])
    print(len(snts_chosen))
    print(len(y_zemberek_chosen))

    toplam = 0
    match = 0
    for i in indexes:
        for j in range(len(y_zemberek[i])):
            toplam = toplam + 1
            if y_my[i][j] == y_zemberek[i][j]:
                match = match + 1

    print("lololoooloololololololo")
    print(match)
    print(toplam)
    print("lololoooloololololololo")

    all_lines = []
    for i in range(len(snts_chosen)):
        all_lines.append("# text = " + snts_chosen[i])
        for j in range(len(y_zemberek_chosen[i])):
            all_lines.append(y_zemberek_chosen[i][j])

    textfile = open("chosen_test_productview_v2.txt", "w", encoding="utf-8")
    for element in all_lines:
        textfile.write(element + "\n")
    textfile.close()

