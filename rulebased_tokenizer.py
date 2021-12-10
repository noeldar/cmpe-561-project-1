import re


def _captured_pattern(pattern):
    return r'(' + pattern + r')'


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



class rulebased_tokenizer:





    # phrase boundary
    ##################################

    PUNCT_END_PHRASE = frozenset(
            [")", "]", "“", "'", "»", "”", "’", '"', "[…]",
             ".", ";", ":", ",", "?", "!", ","])

    # digit, unit
    ###################################

    # Derived unit : (base SI unit)
    si_units = [
            "m²", "fm", "cm²", "m³", "cm³", "l", "ltr", "dl", "cl", "ml",
            "°C", "°F", "K", "g", "gr", "kg", "t", "mg", "μg", "m", "km",
            "mm", "μm", "cm", "sm", "s", "ms", "μs", "Nm", "klst", "min",
            "W", "mW", "kW", "MW", "GW", "TW", "J", "kJ", "MJ", "GJ", "TJ",
            "kWh", "MWh", "kWst", "MWst", "kcal", "cal", "N", "kN", "V", "v",
            "mV", "kV", "A", "mA", "Hz", "kHz", "MHz", "GHz", "Pa", "hPa",
            "°", "°c", "°f"]

    digits_pn = r'(?:\b|^)[-+±~]?(?:\d[-.,0-9\/#]*\d|'\
                    r'\d+(?:st|nd|rd|th|[dD])?)'\
                    r'[%]?(?:\b|$)'
    digits_captured_pn = _captured_pattern(digits_pn)

    DIGIT_RE = re.compile(r'\d')
    DIGITS_RE = re.compile(digits_pn)
    DIGITS_CAPTURED_RE = re.compile(digits_captured_pn)
    YEAR_RE = re.compile(r'(\d+/\d+/\d+)')# (?:\b|^)(?:19|20)\d\d(?:\b|$)
    TIME_RE = re.compile(r'(\d{2}\:\d{2})')

    # url, email
    ##################################
    url_pn = r"(?:[0-9a-zA-Z][-\w_]+)" \
             r"(?:\.[0-9a-zA-Z][-\w_]+){2,5}" \
             r"(?:(?:\/(?:[0-9a-zA-Z]|[-_?.#=:&%])+)+)?\/?"

    url_strict_pn = r'(?:(?:http[s]?|ftp)://|wwww?[.])' \
                    r'(?:[a-zA-Z]|[0-9]|[-_:\/?@.&+=]|' \
                    r'(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    email_pn = r"\S+[@]\S+[.]\S+"
    domain_pn = r"[@]\S+[.]\S+"
    all_web_pn = "|".join([url_strict_pn, url_pn, email_pn, domain_pn])
    all_web_captured_pn = _captured_pattern(all_web_pn)

    URL_RE = re.compile("|".join([url_pn, url_strict_pn]))
    EMAIL_RE = re.compile(email_pn)
    DOMAIN_RE = re.compile(domain_pn)

    ALL_WEB_RE = re.compile(all_web_pn)
    ALL_WEB_CAPTURED_RE = re.compile(all_web_captured_pn)

    PUNCT_SEQ_RE = re.compile(r'[-!\'#%&`()\[\]*+,.\\/:;<=>?@^$_{|}~]+')
    PARA_SEP_RE = re.compile(r'(\W|\+\-)\1{4,}')


    # word bourdary
    ##################################

    word_bf_pn = r'[()\[\]{}"“”\'`»:;,/\\*?!…<=>@^$\|~%]|' \
                 r'[\u2022\u2751\uF000\uF0FF]|' \
                 r'[\u25A0-\u25FF]|' \
                 r'\.{2,}'
    word_bf_captured_pn = _captured_pattern(word_bf_pn)
    WORD_BF_CAPTURED_RE = re.compile(word_bf_captured_pn)

    # looks the same, but actually different hyphens
    hyphen_pn = r'[\-\–\—]'
    HYPHEN_RE = re.compile(hyphen_pn)
    HYPHEN_CAPTURED_RE = re.compile(_captured_pattern(hyphen_pn))
    turkish_abbreviation_set = load_abbreviations("abbreviations.txt")
    regexp = re.compile(r'[^\s]+|\s+')
    space_regexp = re.compile(r'\s')

    def abbreviation(self,phrase):
        return phrase in self.turkish_abbreviation_set

    def _adjust_on_punc(self,token):
            if self.PUNCT_SEQ_RE.fullmatch(token) and self.PARA_SEP_RE.fullmatch(token) is None:
                # a string of punc, very likely .. or ...
                for shift, single_char in enumerate(token):
                    yield single_char

            elif self._has_end_of_phrase_punc(token) and self._phrase_full_match(token) in [None, 'url/email']:
                for splitted_token in [token[:-1],token[-1]]:
                    yield splitted_token
            else:
                yield token

    def _phrase_full_match(self, phrase):
            matched_type = None
            if len(phrase) == 1:
                matched_type = 'single_char'
            elif phrase.isalpha():
                matched_type = 'word'
            elif phrase in self.si_units:
                matched_type = 'unit'
            elif self.DIGITS_RE.fullmatch(phrase):
                matched_type = 'digit'
            elif self.YEAR_RE.fullmatch(phrase):
                matched_type = 'year'
            elif self.TIME_RE.fullmatch(phrase):
                matched_type = 'time'
            elif self.PARA_SEP_RE.fullmatch(phrase):
                matched_type = 'punctuation_seq'
            elif self.abbreviation(phrase):
                matched_type = 'abbreviation'
            elif self.ALL_WEB_RE.fullmatch(phrase):
                matched_type = 'url/email'
            return matched_type


    def _has_end_of_phrase_punc(self,phrase):
        end_char_is_punc = False
        if phrase[-1] in self.PUNCT_END_PHRASE:
            end_char_is_punc = True
            if self.abbreviation(phrase):
                end_char_is_punc = False
        return end_char_is_punc

    def _top_down_tokenize(self,phrase):
            # first get the web url and emails out
            for token in self._top_down_level_1(phrase):
                yield token

    def _top_down_level_1(self,phrase):

            for sub_phrase in re.split(self.ALL_WEB_CAPTURED_RE, phrase):
                if sub_phrase == '':
                    continue
                if self._phrase_full_match(sub_phrase) is not None:
                    yield sub_phrase

                else:
                    for token in self._top_down_level_2(sub_phrase):
                        yield token


    def _top_down_level_2(self,phrase):

            for sub_phrase in re.split(self.DIGITS_CAPTURED_RE, phrase):
                if sub_phrase == '':
                    continue
                if self._phrase_full_match(sub_phrase) is not None:
                    yield sub_phrase
                else:
                    for token in self._top_down_level_3(sub_phrase):
                        yield token

    def _top_down_level_3(self,phrase):
            for sub_phrase in re.split(self.WORD_BF_CAPTURED_RE, phrase):
                if sub_phrase == '':
                    continue
                if self._phrase_full_match(sub_phrase) is not None:
                    yield sub_phrase
                else:
                    for token in self._top_down_level_4(sub_phrase):
                        yield token


    def _top_down_level_4(self,phrase):
            splitted = False
            parts = []
            # - split on hyphen #
            if self.HYPHEN_RE.search(phrase):
                splitted = True
                parts = [ part for part in self.HYPHEN_CAPTURED_RE.split(phrase) if part != '']
                if len(parts) == 3:
                    if len(parts[0]) < 4 and len(parts[2]) < 4 and len(parts[0]) + len(parts[2]) < 6:
                        # mx-doc, tcp-ip, e-mail, hp-ux etc. #
                        splitted = False

            if splitted:
                for part in parts:
                    yield part

            else:
                # pick up what ever left as a token #
                yield phrase

    def _tokenize(self,text):

        for match in self.regexp.finditer(text):
            phrase = match.group()
            if self.space_regexp.search(phrase):
                continue
            if self._phrase_full_match(phrase) is not None:
                for adjusted_token in self._adjust_on_punc(phrase):
                    yield adjusted_token
            else:
                for token in self._top_down_tokenize(phrase):
                    for adjusted_token in self._adjust_on_punc(token):
                        yield adjusted_token

"""
dnm = "Saat 12:00."

a=rulebased_tokenizer()

tokens = a._tokenize(dnm)
for token in tokens:
    print(token)
"""