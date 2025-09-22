"""
Add modules you want to import first
"""
from collections import Counter
from nltk import SnowballStemmer
from nltk.corpus import stopwords


class CorpusReader_SLM:
    def __init__(self, corpus, stopWord='none', toStem=False, smooth=False, trigram=False):
        self.enableSmoothing = smooth
        self.enableTrigram = trigram
        self.enableStemmer = toStem
        self.vocab = set()
        self.unigramCounts = Counter()
        self.bigramCounts = Counter()
        self.trigramCounts = Counter()

        # Handle the stop words or read stop words from a file
        if stopWord == 'standard':
            self.stopwords = set(stopwords.words('english'))
        elif stopWord == 'none':
            self.stopwords = set()
        else:
            with open(stopWord, 'r') as f:
                self.stopwords = set(line.strip().lower() for line in f)

        # Add Stemmer if enabled True
        self.stemmer = SnowballStemmer('english') if toStem else None

        # Process corpus and update unique counts and n-gram counts
        for sentence in corpus.sents():
            processed = []
            for word in sentence:
                word = word.lower()
                if word in self.stopwords:
                    continue
                if self.stemmer:
                    word = self.stemmer.stem(word)
                processed.append(word)

            # Update and save the unique word form the corpus
            self.vocab.update(processed)

            # Update the unigram counts
            self.unigramCounts.update(processed)

            # Create bi-gram tuple and update count
            for i in range(len(processed) - 1):
                bigram_tuple = (processed[i], processed[i + 1])
                self.bigramCounts[bigram_tuple] += 1

            # Create tri-gram tuple if tri-gram flag is True
            if trigram:
                for i in range(len(processed) - 2):
                    trigram_tuple = (processed[i], processed[i + 1], processed[i + 2])
                    self.trigramCounts[trigram_tuple] += 1

        # Precompute sum totals from Counter() method to calculate probabilities
        self.totalUnigramCounts = sum(self.unigramCounts.values())
        self.totalBigramCounts = sum(self.bigramCounts.values())
        self.totalTrigramCounts = sum(self.trigramCounts.values())

    def unigram(self, count=0):

        return []

    def bigram(self, count=0):
        return []

    def trigram(self, count=0):
        return []

    def unigramGenerate(self, code=0, head=[]):
        return ""

    def bigramGenerate(self, code=0, head=[]):
        return ""

    def trigramGenerate(self, code=0, head=[]):
        return ""
