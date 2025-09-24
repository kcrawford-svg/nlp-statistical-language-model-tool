import nltk
from nltk.corpus import PlaintextCorpusReader, treebank
from CorpusReader_SLM import *

'''
print(len(inaugural.words()))
print(inaugural.sents())
print(len(inaugural.sents()))
print(inaugural.fileids())
print(inaugural.sents(['1789-washington.txt']))

print(len(brown.words()))
print(brown.sents())
'''
'''
print("Treebank corpus stats: ")
print(len(treebank.words()))
print(treebank.sents())
'''
tree_corpus_model = CorpusReader_SLM(treebank, stopWord='none', toStem=False, smooth=True, trigram=True)

# Test unigram, bigram, trigram probabilities
print("\nUnigram Word Samples")
for word, prob in tree_corpus_model.unigram(count=0):
    print(f"{word}\t{prob:.4f}")

print("\nUnigram Word Samples")
for word, prob in tree_corpus_model.unigram(count=0):
    print(f"{word}\t{prob:.4f}")

print("\nBigram Word Samples")
for word, prob in tree_corpus_model.bigram(count=10):
    print(f"{word}\t{prob:.4f}")

print("\nTrigram Word Samples")
for word, prob in tree_corpus_model.trigram(count=10):
    print(f"{word}\t{prob:.4f}")

'''
print("\nGenerated Sentences using Unigram Model:")
print(tree_corpus_model.unigramGenerate(code=1))

print("\nGenerated Sentences using Bigram Model:")
print(tree_corpus_model.bigramGenerate(code=2))
'''
print("\nGenerated Sentences using Trigram Model:")
print(tree_corpus_model.trigramGenerate(code=2))


'''
myCorpus = CorpusReader_SLM(inaugural)

print(myCorpus.unigram())
print(myCorpus.bigram())
print(myCorpus.unigramGenerate())
print(myCorpus.unigramGenerate(10, [This]))

'''
'''
#  This is for testing your own corpus
#
#  create a set of text files, store them in a directory specified from 'rootDir' variable
#
#  

rootDir = '/Users/keenancrawford/PycharmProjects/nlp-statistical-language-model/_temp1'   # change that to the directory where the files are

newCorpus = PlaintextCorpusReader(rootDir, '.*txt')
x = newCorpus.sents()
for y in x:
    print(y)
'''

'''
myC2 = CorpusReader_TFIDF(newCorpus)


print("-----\n")

'''
