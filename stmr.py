import re

def Suffix(pattern, letter, word):
    # letter: kaynaştırma harfi
    # pattern: ekler
    pattern = re.compile('('+pattern+')$', re.U)
    
    if letter is None:
        letterCheck = False
        letterPattern = None
    else:
        letterCheck = True
        letterPattern = re.compile('('+letter+')$', re.U)
    
    if letterCheck:
        match = letterPattern.search(word)
        if match:
            word = match.group()
    
    word = pattern.sub('',word)
    return word

# transition: class object
def AddTransitions(word, transitions, marked, suffixes):
    for suffix in suffixes:
        if suffix.Match(word):
            transitions.append(NextState(suffix, word, marked))

# 2 of the same name function
def NextState(suffix):
    raise NotImplementedError("Feature is not implemented.")

def similarTransitions(transitions,startState,nextState,word,marked):
    for transition in transitions:
        if (startState == transition.startState and 
            nextState == transition.nextState):
            yield transition


# suffix functions according to word type
def derivational(word):
    S1 = Suffix("lı|li|lu|lü", None, word)
    VALUES = (S1)
    return VALUES

def verb(word):
    S11 = Suffix("casına|çasına|cesine|çesine",None, word)
    S4  = Suffix("sınız|siniz|sunuz|sünüz", None, word)
    S14 = Suffix("muş|miş|müş|mış","y", word)
    S15 = Suffix("ken","y", word )
    S2  = Suffix("sın|sin|sun|sün",None, word)
    S5  = Suffix("lar|ler",None, word)
    S9  = Suffix("nız|niz|nuz|nüz",None, word)
    S10 = Suffix("tır|tir|tur|tür|dır|dir|dur|dür",None,word)
    S3  = Suffix("ız|iz|uz|üz","y",word)
    S1  = Suffix("ım|im|um|üm","y",word)
    S12 = Suffix("dı|di|du|dü|tı|ti|tu|tü","y",word)
    S13 = Suffix("sa|se","y",word)
    S6  = Suffix("m",None, word)
    S7  = Suffix("n",None, word)
    S8  = Suffix("k",None, word)

    # The order of the enum definition determines the priority of the suffix.
    # For example, -(y)ken (S15 suffix) is  checked before -n (S7 suffix).
    VALUES = (S11,S4,S14,S15,S2,S5,S9,S10,S3,S1,S12,S13,S6,S7,S8)
    return min(VALUES)

def noun(word):
    S16 = Suffix("ndan|ntan|nden|nten",None,word)
    S7  = Suffix("ları|leri",None,word)
    S3  = Suffix("mız|miz|muz|müz","ı|i|u|ü",word) 
    S5  = Suffix("nız|niz|nuz|nüz","ı|i|u|ü",word) 
    S1  = Suffix("lar|ler",None,word)
    S14 = Suffix("nta|nte|nda|nde",None,word)
    S15 = Suffix("dan|tan|den|ten",None,word)
    S17 = Suffix("la|le","y", word)
    S10 = Suffix("ın|in|un|ün","n",word)
    S19 = Suffix("ca|ce","n",word)
    S4  = Suffix("ın|in|un|ün",None,word)
    S9  = Suffix("nı|ni|nu|nü",None,word) 
    S12 = Suffix("na|ne",None,word)
    S13 = Suffix("da|de|ta|te",None,word)
    S18 = Suffix("ki",None,word)
    S2  = Suffix("m","ı|i|u|ü", word)
    S6  = Suffix("ı|i|u|ü","s",word)
    S8  = Suffix("ı|i|u|ü","y",word)
    S11 = Suffix("a|e","y",word)

    # The order of the enum definition determines the priority of the suffix.
    # For example, -(y)ken (S15 suffix) is  checked before -n (S7 suffix).
    VALUES = (S16,S7,S3,S5,S1,S14,S15,S17,S10,S19,S4,S9,S12,S13,S18,S2,S6,S8,S11)
    return min(VALUES)
