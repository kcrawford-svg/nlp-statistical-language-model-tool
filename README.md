# NLP Statistical Language Model


A Python implementation of an unigram/bigram/trigram **Statistical Language Model (SLM)**  
for the Introduction to Natural Language Processing course.


## Features
- Builds unigram, bigram, and optional trigram probability models
- Supports Laplace smoothing, stop-word removal, and stemming
- Generates text using unigram/bigram/trigram probabilities

## Setup
```bash
conda create -n nlp-slm python=3.11
conda activate nlp-slm
conda install nltk

Can also use pip install nltk is not using conda