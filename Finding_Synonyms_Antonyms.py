# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 00:14:23 2018

@author: Srinivas
"""

# Finding synonyms and Antonyms of the word
from nltk.corpus import wordnet

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for s in syn.lemmas():
        synonyms.append(s.name())
        for a in s.antonyms():
            antonyms.append(a.name())
            
print(set(synonyms)) # We are taking sets to remove the duplicates
print(set(antonyms))    
