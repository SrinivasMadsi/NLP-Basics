# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 19:14:42 2018

@author: Srinivas
"""

# Creating an Article Summarizer
import  bs4 as bs
import urllib.request
import re
import nltk
nltk.download('stopwords')
import heapq


# Getting the data
source = urllib.request.urlopen('https://en.wikipedia.org/wiki/Global_warming').read()  

# Now the data from wikipedia is in html format, we need to parse the data

soup = bs.BeautifulSoup(source,'lxml')
text = ""
for paragraph in soup.find_all('p'): # In wikipedia Everything is in the Paragraph tag only, so we extract the content from 'p'
    text += paragraph.text

# Pre-processing the extracted text
text = re.sub(r'\[[0-9]*\]',"",text)    # To remove the references from wikipedia text
text = re.sub(r'\s+'," ",text) # Replacing extra spaces with single space
clean_text = text.lower()
clean_text = re.sub(r'\W',' ',clean_text) # Removing the non-words
clean_text = re.sub(r'\d',' ',clean_text) # Removing the digits
clean_text = re.sub(r'\s+',' ',clean_text)

# Now we need to draw histogram, 
# we need to do the tokenization and remove the stop words
sentences = nltk.sent_tokenize(text) # clean_text is already cleaned so we cannot frame sentences from it, so we use text
stop_words = nltk.corpus.stopwords.words('english')

word2count = {}
for word in nltk.word_tokenize(clean_text):
    if word not in stop_words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1 # We can draw basic histogram with this
            
    
# Now we try to create weightage histogram for that we need word/max value
for key in word2count.keys():
    word2count[key] = word2count[key]/max(word2count.values())   

# Now we will find the sentences score
sent2score = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word2count.keys():
            if len(sentence.split(' ')) <25 :   # Sentences >25 size are not much useful
                if sentence not in sent2score.keys():
                    sent2score[sentence] = word2count[word] 
                else:
                    sent2score[sentence] += word2count[word]
    
# Now we will find the best sentences to display the summary
best_sentences = heapq.nlargest(5,sent2score,key=sent2score.get)                    

print('-------------------------------------------------------------------------')
for sentence in best_sentences:
    print(sentence)



         
 