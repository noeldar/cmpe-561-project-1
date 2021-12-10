from rulebased_tokenizer import *
from typing import List, Tuple, Dict, FrozenSet, Set, Union
import numpy as np
import pandas as pd
from zemberek.normalization.character_graph_decoder import CharacterGraphDecoder
from zemberek.normalization.deasciifier.deasciifier import Deasciifier
from zemberek.normalization.stem_ending_graph import StemEndingGraph
from zemberek.morphology import TurkishMorphology
from MorphologicalAnalysis.FsmMorphologicalAnalyzer import FsmMorphologicalAnalyzer
from MorphologicalAnalysis.State import State
from MorphologicalAnalysis.Transition import Transition

fsm = FsmMorphologicalAnalyzer()
"""word = "yran"
fsmParseList = fsm.morphologicalAnalysis(word)
for i in range(fsmParseList.size()):
  	print(fsmParseList.getFsmParse(i).transitionList())"""

morphology = TurkishMorphology.create_with_defaults()
graph = StemEndingGraph(morphology)
decoder = CharacterGraphDecoder(graph.stem_graph)

#print(decoder.get_suggestions("yran", CharacterGraphDecoder.DIACRITICS_IGNORING_MATCHER))

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

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    return data

def tokenize(sentence):
    rt = rulebased_tokenizer()
    tokens = rt._tokenize(sentence)
    tmp = []
    for token in tokens:
        tmp.append(token)
    return tmp

def load_replacements(filename):
    with open(filename, "r",
              encoding="utf-8") as f:
        replacements: Dict[str, str] = {}
        for line in f:
            tokens = line.replace('\n', "").split("=")
            replacements[tokens[0].strip()] = tokens[1].strip()
    return replacements


def load_no_split(filename):
    with open(filename, "r", encoding="utf-8") as f:
        s = set()
        for line in f:
            if len(line.replace('\n', "").strip()) > 0:
                s.add(line.replace('\n', "").strip())
    return frozenset(s)


def load_common_split(filename):
    common_splits: Dict[str, str] = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.replace('\n', "").split('-')
            common_splits[tokens[0].strip()] = tokens[1].strip()
    return common_splits


def load_multimap(resource):
    with open(resource, "r", encoding="utf-8") as f:
        lines: List[str] = f.read().split('\n')
    multimap: Dict[str, Tuple[str, ...]] = {}
    for i, line in enumerate(lines):
        if len(line.strip()) == 0:
            continue
        index = line.find("=")
        if index < 0:
            raise BaseException(f"Line needs to have `=` symbol. But it is: {i} -" + line)
        key, value = line[0:index].strip(), line[index + 1:].strip()
        if value.find(',') >= 0:
            if key in multimap.keys():
                multimap[key] = tuple(value.split(','))
        else:
            if key in multimap.keys():
                multimap[key] = multimap[key] + (value,)
            else:
                multimap[key] = (value,)
    return multimap

def populate_dict(dictionary, in_str, out_str):
        for in_, out in zip(in_str, out_str):
            dictionary[in_] = out

        return dictionary


with open("question-suffixes.txt", "r",encoding="utf-8") as f:
    lines = f.read().split('\n')

common_connected_suffixes = frozenset(lines)
ins_cost = 1
del_cost = 1
sub_cost = 2
replacements = load_replacements("multi-word-replacements.txt")
no_split_words = load_no_split("no-split.txt")
common_splits = load_common_split("split.txt")
lookup_manual = load_multimap("candidates-manual.txt")
lookup_from_graph = load_multimap("lookup-from-graph.txt")
lookup_from_ascii = load_multimap("ascii-map.txt")


def min_cost_path(cost, operations):
    # operation at the last cell
    path = [operations[cost.shape[0] - 1][cost.shape[1] - 1]]

    # cost at the last cell
    min_cost = cost[cost.shape[0] - 1][cost.shape[1] - 1]

    row = cost.shape[0] - 1
    col = cost.shape[1] - 1

    while row > 0 and col > 0:

        if cost[row - 1][col - 1] <= cost[row - 1][col] and cost[row - 1][col - 1] <= cost[row][col - 1]:
            path.append(operations[row - 1][col - 1])
            row -= 1
            col -= 1
        elif cost[row - 1][col] <= cost[row - 1][col - 1] and cost[row - 1][col] <= cost[row][col - 1]:
            path.append(operations[row - 1][col])
            row -= 1
        else:
            path.append(operations[row][col - 1])
            col -= 1

    return "".join(path[::-1][1:])


def edit_distance_dp(seq1, seq2):
    # create an empty 2D matrix to store cost
    cost = np.zeros((len(seq1) + 1, len(seq2) + 1))

    # fill the first row
    cost[0] = [i for i in range(len(seq2) + 1)]

    # fill the first column
    cost[:, 0] = [i for i in range(len(seq1) + 1)]

    # to store the operations made
    operations = np.asarray([['-' for j in range(len(seq2) + 1)] \
                             for i in range(len(seq1) + 1)])

    # fill the first row by insertion
    operations[0] = ['I' for i in range(len(seq2) + 1)]

    # fill the first column by insertion operation (D)
    operations[:, 0] = ['D' for i in range(len(seq1) + 1)]

    operations[0, 0] = '-'

    # now, iterate over earch row and column
    for row in range(1, len(seq1) + 1):

        for col in range(1, len(seq2) + 1):

            # if both the characters are same then the cost will be same as
            # the cost of the previous sub-sequence
            if seq1[row - 1] == seq2[col - 1]:
                cost[row][col] = cost[row - 1][col - 1]
            else:

                insertion_cost = cost[row][col - 1] + ins_cost
                deletion_cost = cost[row - 1][col] + del_cost
                substitution_cost = cost[row - 1][col - 1] + sub_cost

                # calculate the minimum cost
                cost[row][col] = min(insertion_cost, deletion_cost, substitution_cost)

                # get the operation
                if cost[row][col] == substitution_cost:
                    operations[row][col] = 'S'

                elif cost[row][col] == ins_cost:
                    operations[row][col] = 'I'
                else:
                    operations[row][col] = 'D'

    return cost[len(seq1), len(seq2)], min_cost_path(cost, operations)


def replace_common(tokens):
    result = []
    for token in tokens:
        text = token
        result.append(replacements.get(text, text))
    return ' '.join(result)

def combine_necessary_words(tokens):
        result = []
        combined = False
        for i in range(len(tokens) - 1):
            first_s = tokens[i]
            second_s = tokens[i + 1]
            w1 = fsm.morphologicalAnalysis(first_s)
            w2 = fsm.morphologicalAnalysis(second_s)

            if w1.size() > 0 and w2.size() > 0:
                if combined:
                    combined = False
                else:
                    c = combine_common(first_s, second_s)
                    if len(c) > 0:
                        result.append(c)
                        combined = True
                    else:
                        result.append(first_s)
                        combined = False
            else:
                combined = False
                result.append(first_s)

        if not combined:
            result.append(tokens[-1])
        return ' '.join(result)

def combine_common(i1, i2):
        combined = i1 + i2
        w = fsm.morphologicalAnalysis(combined)
        w1 = fsm.morphologicalAnalysis(i2)
        if i2.startswith("'") or i2.startswith("bil"):

            if w.size() > 0:
                return combined

        if w1.size() <= 0:

            if w.size() > 0:
                return combined
        return ""

def split_necessary_words(tokens, use_look_up):
        result = []

        for token in tokens:
            text = token
            w = fsm.morphologicalAnalysis(token)
            if w.size() > 0:
                result.append(separate_common(text, use_look_up))
            else:
                result.append(text)

        return ' '.join(result)

def separate_common(inp, use_look_up):
        if inp in no_split_words:
            return inp
        if use_look_up and inp in common_splits:
            return common_splits[inp]
        w = fsm.morphologicalAnalysis(inp)
        if w.size() <= 0:
            for i in range(len(inp)):
                tail = inp[i:]
                if tail in common_connected_suffixes:
                    head = inp[0:i]
                    w1 = fsm.morphologicalAnalysis(head)
                    if w1.size() > 0:
                        return f"{head} {tail}"
                    else:
                        return inp
        return inp

def probably_requires_deasciifier(sentence):
        turkish_spec_count = 0
        for c in sentence:
            if c != 'ı' and c != 'I' and c in set("çÇğĞıİöÖşŞüÜâîûÂÎÛ"):
                turkish_spec_count += 1
        ratio = turkish_spec_count * 1. / len(sentence)
        return ratio < 0.1

def preprocess(data):
    data = convert_lower_case(data)
    tokens = tokenize(data)
    s = replace_common(tokens)
    tokens = tokenize(s)
    s = combine_necessary_words(tokens)
    tokens = tokenize(s)
    s = split_necessary_words(tokens, use_look_up=False)
    if probably_requires_deasciifier(s):
        deasciifier = Deasciifier(s)
        s = deasciifier.convert_to_turkish()

    tokens = tokenize(s)
    s = combine_necessary_words(tokens)
    tokens = tokenize(s)
    return split_necessary_words(tokens, use_look_up=True)




def normalize(sentence):
    sentence = convert_lower_case(sentence)
    sentence = preprocess(sentence)
    #print(sentence)
    tokens = tokenize(sentence)


    for i, current_token in enumerate(tokens):
        current = current_token
        next_ = None if i == len(tokens) - 1 else tokens[i + 1]
        previous = None if i == 0 else tokens[i - 1]

        fsmParseList = fsm.morphologicalAnalysis(current)

        if fsmParseList.size() == 0:

            candidates = set()

            candidates.update(lookup_manual.get(current, ()))
            candidates.update(lookup_from_graph.get(current, ()))
            candidates.update(lookup_from_ascii.get(current, ()))
            if len(candidates) <= 0:
                sugs = decoder.get_suggestions(current, CharacterGraphDecoder.DIACRITICS_IGNORING_MATCHER)
                if len(sugs) > 3:
                    candidates.update(set(sugs[:500]))
                    #for ss in sugs[:3]:
                else:
                    candidates.update(set(sugs))





            min_cost=100000
            min_cand=""

            for cand in candidates:
                score, operations = edit_distance_dp(current, cand)
                if score < min_cost:
                    min_cost = score
                    min_cand = cand

            #print(min_cand+" "+str(min_cost))
            if min_cost<1000:
                tokens[i]=min_cand

    return ' '.join(tokens)





from zemberek import (
    TurkishSpellChecker,
    TurkishSentenceNormalizer,
    TurkishSentenceExtractor,
    TurkishMorphology,
    TurkishTokenizer
)
morphology = TurkishMorphology.create_with_defaults()
aa=[]
aa_norm=[]
normalizer = TurkishSentenceNormalizer(morphology)
with open("norm_data.txt","r", encoding="utf-8") as file:
    for line in file:
        #print(line)
        hel=normalizer.normalize(line)
        if line!=hel:
            aa.append(line)
            aa_norm.append(hel)

bb=[]
bb_norm=[]
with open("norm_data.txt","r", encoding="utf-8") as file:
    for line in file:
        #print(line)
        hel=normalize(line)
        if line!=hel:
            bb.append(line)
            bb_norm.append(hel)

print(len(list(aa)))
print(len(list(bb)))
print(len(list(set(aa) & set(bb))))
####################################################
print(len(list(set(aa_norm) & set(bb_norm))))

