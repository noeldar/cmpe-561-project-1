import re

def load_abbreviations(filename):
    """
    function that loads Turkish abbreviations from a text file on given path It stores both
    original and lower cased version in a set

    :param path: text file that contains abbreviations as one abbreviation per line
    :return: set of strings storing both original and lower cased abbreviations
    """
    lower_map = {
        ord(u'I'): u'ı',
        ord(u'İ'): u'i',
    }

    abbr_set = set()
    with open(filename, 'r', encoding="utf-8") as f:
        lines = list(f.readlines())
        for line in lines:
            if len(line.strip()) > 0:
                abbr = re.sub(r'\s+', "", line.strip())
                abbr_set.add(re.sub(r'\.$', "", abbr))
                abbr = abbr.translate(lower_map)
                abbr_set.add(re.sub(r'\.$', "", abbr.lower()))

    return abbr_set




class rulebased_sentencesplitter:
    SENT_RE = re.compile(r'[^\.?!…]+[\.?!…]*["»“]*')
    _LAST_WORD = re.compile(r'(?:\b|\d)([a-zğçşöüı]+)\.$', re.IGNORECASE)
    _FIRST_WORD = re.compile(r'^\W*(\w+)')
    _ENDS_WITH_ONE_LETTER_LAT_AND_DOT = re.compile(r'(\d|\W|\b)([a-zA-Z])\.$')
    _HAS_DOT_INSIDE = re.compile(r'[\w]+\.[\w]+\.$', re.IGNORECASE)
    _INITIALS = re.compile(r'(\W|\b)([A-ZĞÇŞÖÜİ]{1})\.$')
    _STARTS_WITH_EMPTYNESS = re.compile(r'^\s+')
    _ENDS_WITH_EMOTION = re.compile(r'[!?…]|\.{2,}\s?[)"«»,“]?$')
    _STARTS_WITH_LOWER = re.compile(r'^\s*[–-—-("«]?\s*[a-zğçşöüı]')
    _STARTS_WITH_DIGIT = re.compile(r'^\s*\d')
    _NUMERATION = re.compile(r'^\W*[IVXMCL\d]+\.$')
    _PAIRED_SHORTENING_IN_THE_END = re.compile(r'\b(\w+)\. (\w+)\.\W*$')

    _JOIN = 0
    _MAYBE = 1
    _SPLIT = 2




    turkish_abbreviation_set = load_abbreviations("abbreviations.txt")

    def _regex_split_separators(self,text):
        return [x.strip() for x in self.SENT_RE.findall(text)]


    def _is_sentence_end(self, left, right, shortenings):
        if not self._STARTS_WITH_EMPTYNESS.match(right):
            return self._JOIN

        if self._HAS_DOT_INSIDE.search(left):
            return self._JOIN

        left_last_word = self._LAST_WORD.search(left)
        lw = ' '
        if left_last_word:
            lw = left_last_word.group(1)

        if self._ENDS_WITH_EMOTION.search(left) and self._STARTS_WITH_LOWER.match(right):
            return self._JOIN

        initials = self._INITIALS.search(left)
        if initials:
            border, _ = initials.groups()
            if (border or ' ') not in "°'":
                return self._JOIN

        if lw.lower() in shortenings:
            return self._MAYBE

        last_letter = self._ENDS_WITH_ONE_LETTER_LAT_AND_DOT.search(left)
        if last_letter:
            border, _ = last_letter.groups()
            if (border or ' ') not in "°'":
                return self._MAYBE
        if self._NUMERATION.match(left):
            return self._JOIN
        return self._SPLIT

    def sent_tokenize(self,text, shortenings):
        sentences = []
        sents = self._regex_split_separators(text)
        si = 0
        processed_index = 0
        sent_start = 0
        while si < len(sents):
            s = sents[si]
            span_start = text[processed_index:].index(s) + processed_index
            span_end = span_start + len(s)
            processed_index += len(s)

            si += 1

            send = self._is_sentence_end(text[sent_start: span_end], text[span_end:], shortenings)
            if send == self._JOIN:
                continue

            if send == self._MAYBE:
                if self._STARTS_WITH_LOWER.match(text[span_end:]):
                    continue
                if self._STARTS_WITH_DIGIT.match(text[span_end:]):
                    continue

            if not text[sent_start: span_end].strip():
                print("Something went wrong while tokenizing")


            sentences.append(text[sent_start: span_end].strip())
            sent_start = span_end
            processed_index = span_end

        if sent_start != len(text):
            if text[sent_start:].strip():
                sentences.append(text[sent_start:].strip())
        return sentences


aa = "Hâla AMR istediğinden eminim ."
bb=rulebased_sentencesplitter()
print(bb.sent_tokenize(aa, bb.turkish_abbreviation_set))