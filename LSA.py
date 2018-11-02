# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:17:43 2018

@author: Srinivas
"""

# Latent Semantic Analysis using Python [Finding the concepts from documents]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk

dataset = ["Thank you all so very much. Thank you to the Academy",
               "Thank you to all of you in this room",
               "I have to congratulate the other incredible nominees this year",
               "The Revenant was the product of the tireless efforts of an unbelievable cast  and crew",
               "First off, to my brother in this endeavor, Mr. Tom Hardy.",
               "Tom, your talent on screen can only be surpassed by our friendship off screen"]
dataset = [line.lower() for line in dataset]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset) # We are converting to  TF-IDF model because it has many features
lsa = TruncatedSVD(n_components=4,n_iter =100)
lsa.fit(X)               
row1 = lsa.components_[0]
concept_words = {}

terms = vectorizer.get_feature_names()
for i,comp in enumerate(lsa.components_):
    componentTerms = zip(terms,comp)
    sortedTerms = sorted(componentTerms,key = lambda x:x[1],reverse = True)
    sortedTerms= sortedTerms[:10]
    concept_words["Concept " + str(i)] = sortedTerms

for key in concept_words.keys():
    sentence_scores = []
    for sentence in dataset:
        words = nltk.word_tokenize(sentence)
        score = 0
        for word in words:
            for word_with_score in concept_words[key]:
                if word == word_with_score[0]:
                    score += word_with_score[1]
        sentence_scores.append(score)
    print("\n"+key+":")
    for sentence_score in sentence_scores:
        print(sentence_score)           














