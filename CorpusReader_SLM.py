"""
Add modules you want to import first
"""
import random
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
        self.bigram_cond_counts = Counter()
        self.trigram_cond_counts = Counter()

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
            if not processed:
                continue

            # Update and save the unique word form the corpus
            self.vocab.update(processed)

            # Update the unigram counts
            self.unigramCounts.update(processed)

            # Create bi-gram tuple and update count
            for i in range(len(processed) - 1):
                word1, word2 = processed[i], processed[i + 1]
                self.bigramCounts[(word1, word2)] += 1
                self.bigram_cond_counts[word1] += 1  # get the first word

            # Create tri-gram tuple if tri-gram flag is True
            if trigram:
                for i in range(len(processed) - 2):
                    word1, word2, word3 = processed[i], processed[i + 1], processed[i + 2]
                    self.trigramCounts[(word1,word2,word3)] += 1
                    self.trigram_cond_counts[(word1, word2)] += 1

        # Precompute sum totals from Counter() method to calculate probabilities
        self.totalUnigramCounts = sum(self.unigramCounts.values())

    def unigram(self, count=0):
        V = len(self.vocab)
        probs = {}
        for word, c in self.unigramCounts.items():
            if self.enableSmoothing:
                probs[word] = (c + 1) / (self.totalUnigramCounts + V)
            else:
                probs[word] = c / self.totalUnigramCounts

        if count == 0:
            return sorted(probs.items(), key=lambda x: x[0])
        else:
            sorted_probs = sorted(probs.items(), key=lambda x: (-x[1], x[0]))
            cutoff_index = sorted_probs[-1][1] if count <= len(sorted_probs) else sorted_probs[-1][1]
            top_ties = [word for word in sorted_probs if word[1] >= cutoff_index]
            return sorted(top_ties, key=lambda x: x[0])


    def bigram(self, count=0):
        prob_dict = {}
        v = len(self.vocab)
        for (word_one, word_two), c in self.bigramCounts.items():
            cond_count = self.bigram_cond_counts[word_one]
            if self.enableSmoothing:
                prob_dict[f'{word_one} {word_two}'] = (c + 1) / (cond_count + v)
            else:
                if cond_count == 0:
                    continue
                return sorted(prob_dict.items(), key=lambda x: x[0])
        if count == 0:
            # Return all bigrams alphabetically
            return sorted(prob_dict.items(), key=lambda x: x[0])
        else:
            # Sort by probability descending first, then alphabetically for ties
            sorted_by_prob = sorted(prob_dict.items(), key=lambda x: (-x[1], x[0]))
            cutoff_index = sorted_by_prob[count - 1][1] if count <= len(sorted_by_prob) else sorted_by_prob[-1][1]
            tied_top = [word for word in sorted_by_prob if word[1] >= cutoff_index]
            return sorted(tied_top, key=lambda x: x[0])

    def trigram(self, count=0):
        if not self.enableTrigram:
            return []

        prob_dict = {}
        v = len(self.vocab)
        for (word_one, word_two, word_three), c in self.trigramCounts.items():
            cond_count = self.trigram_cond_counts.get((word_one, word_two), 0)
            if cond_count == 0:
                continue
            if self.enableSmoothing:
                prob_dict[f"{word_one} {word_two} {word_three}"] = (c + 1) / (cond_count + v)
            else:
                prob_dict[f"{word_one} {word_two} {word_three}"] = c / cond_count
        if count == 0:
            return sorted(prob_dict.items(), key=lambda x: x[0])
        else:
            # Sorted probabilities
            sorted_by_prob = sorted(prob_dict.items(), key=lambda x: (-x[1], x[0]))
            # Get the last probability value in the dict else get the lowest if count = 0
            cutoff_index = sorted_by_prob[count - 1][1] if count <= len(sorted_by_prob) else sorted_by_prob[-1][1]
            tied_top = [word for word in sorted_by_prob if word[1] >= cutoff_index]
            return sorted(tied_top, key=lambda x: x[0])

    def unigramGenerate(self, code=0, head=[]):
        if code not in [0, 1, 2]:
            return ""

        # Get unigram probabilities
        list_of_prob = self.unigram()
        # Generated Probability dictionary
        probDict = {w: p for w, p in list_of_prob}
        sentence = list(head)

        while True:
            nextWord = None
            if code == 0:
                # pick the word with the highest probability
                maxProb = max(probDict.values())
                # list of candidate words if max probability is tied choose random
                candidate = [w for w, p in probDict.items() if p == maxProb]
                nextWord = random.choice(candidate)
            elif code == 1:
                # using probability as weight for word selection
                word, weight = zip(*probDict.items())  # unpack into tuples
                nextWord = random.choices(word, weights=weight, k=1)[0]
            elif code == 2:
                #  top 10 unigram probability
                sort_probs = sorted(probDict.items(), key=lambda x: x[1], reverse=True)
                probIndex = sort_probs[9][1] if len(sort_probs) > 9 else sort_probs[-1][1]
                topWords = [(w, p) for w, p in sort_probs if p >= probIndex]
                word, weight = zip(*topWords)  # unpack into tuples

                # Normalize the probabilities
                total = sum(weight)
                weight = [w / total for w in weight]
                nextWord = random.choices(word, weights=weight, k=1)[0]
            sentence.append(nextWord)

            if nextWord in {".", "!", "?"}:
                break

        output = sentence[0]
        for word in sentence[1:]:
            if word in {".", ",", "!", "?"}:
                output += word
            else:
                output += " " + word
        return output

    def bigramGenerate(self, code=0, head=[]):
        if code not in [0, 1, 2] or not self.bigramCounts:
            return ""

        # Get unigram probabilities
        prob = self.bigram()
        # Generated Probability dictionary
        probDict = {w: p for w, p in prob}

        sentence = []
        if head:
            sentence.extend(head)
        elif self.vocab:
            sentence.append(random.choice(list(self.vocab)))

        nextWord = None
        while True:
            lastWord = sentence[-1]
            # Bigrams that start with the last word dict filter
            candidate = {k.split()[1]: p for k, p in self.bigram(count=0) if k.split()[0] == lastWord}
            if not candidate:
                break

            if code == 0:
                # pick the word with the highest probability
                maxProb = max(candidate.values())
                topWord = [w for w, p in candidate.items() if p == maxProb]
                nextWord = random.choice(topWord)
            elif code == 1:
                # using probability as weight for word selection
                words, weight = zip(*candidate.items())  # word/weight tuples
                nextWord = random.choices(words, weights=weight, k=1)[0]
            elif code == 2:
                #  top 10 unigram probability
                sort_candidates = sorted(probDict.items(), key=lambda x: -x[1])
                probCount = sort_candidates[9][1] if len(sort_candidates) > 9 else sort_candidates[-1][1]
                topCandidate = [(w, p) for w, p in sort_candidates if p >= probCount]
                words, weight = zip(*topCandidate)  # word/weight tuples
                # Normalize the probabilities
                total = sum(weight)
                normalize = [w / total for w in weight]
                nextWord = random.choices(words, weights=normalize, k=1)[0]
            sentence.append(nextWord)

            if nextWord in {".", "!", "?"}:
                break

            output = sentence[0]
            for word in sentence[1:]:
                if word in {".", ",", "!", "?"}:
                    output += word
                else:
                    output += " " + word
            return output

    def trigramGenerate(self, code=0, head=[]):
        if not self.enableTrigram or code not in [0, 1, 2]:
            return ""

        sentence = list(head)
        # Choose at least two words from the vocabulary to initialize
        if len(sentence) < 2:
            if self.trigramCounts:
                # pick a trigram from counts
                w1, w2, w3 = random.choice(list(self.trigramCounts.keys()))
                sentence = [w1, w2]
            else:
                # pick two random words from vocab
                sentence = random.sample(self.vocab, 2)

        while True:
            nextWord = None
            lastTwoWords = sentence[-2:]
            lastPrefix = " ".join(lastTwoWords)

            # Trigrams that start with the last word dict filter
            candidate = {k.split()[2]: p for k, p in self.trigram(count=0) if k.startswith(lastPrefix + " ")}
            if not candidate:
                break

            if code == 0:
                # pick the word with the highest probability
                maxProb = max(candidate.values())
                topWord = [w for w, p in candidate.items() if p == maxProb]
                nextWord = random.choice(topWord)
            elif code == 1:
                # using probability as weight for word selection
                words, weight = zip(*candidate.items())  # word/weight tuples
                nextWord = random.choices(words, weights=weight, k=1)[0]
            elif code == 2:
                #  top 10 unigram probability
                sort_candidates = sorted(candidate.items(), key=lambda x: -x[1])
                probCount = sort_candidates[9][1] if len(sort_candidates) > 9 else sort_candidates[-1][1]
                topCandidate = [(w, p) for w, p in sort_candidates if p >= probCount]
                words, weight = zip(*topCandidate)  # word/weight tuples
                # Normalize the probabilities
                total = sum(weight)
                weight = [w / total for w in weight]
                nextWord = random.choices(words, weights=weight, k=1)[0]
            sentence.append(nextWord)

            if nextWord in {".", "!", "?"}:
                break

            output = sentence[0]
            for word in sentence[1:]:
                if word in {".", ",", "!", "?"}:
                    output += word
                else:
                    output += " " + word
            return output
