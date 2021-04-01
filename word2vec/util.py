import nltk, enchant
with open("words_alpha.txt") as f:
    english_dictionary = set(f.read().split())
tok = nltk.tokenize.toktok.ToktokTokenizer()
stemmer = nltk.stem.porter.PorterStemmer()
def clean_line(line):
    """
        line : untreated string that is to be parsed

        returns:
            - iterable of words that are the individual treated words from the given text
    """
    # get individual tokens and remove punctuation
    return [word for word in tok.tokenize(line.strip().lower()) if word.isalpha() and word in english_dictionary]