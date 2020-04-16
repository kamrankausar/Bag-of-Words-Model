#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip list


# In[1]:


import pandas as pd
import re
#import nltk
# creating the feature matrix 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
#nltk.download('punkt')
#nltk.download('stopwords')


# In[2]:


#df = pd.read_csv('comp_pf_npf_data.csv')
#df_test = df['sen']
#all_sen = list(df_test)


# In[10]:


def lower_remove_words(all_sen):
    clean_sents = []
    for i in range(len(all_sen)):
        #print(all_sen[i])
        if all_sen[i]:
            sen = str(all_sen[i]).lower()
            sen = re.sub('[^a-zA-z0-9]',' ',sen)
            clean_sents.append(sen)
            #print(sen)
    return clean_sents


# In[6]:


def stemming(tokenized_words):
    for i in range(len(tokenized_words)):
        tokenized_words[i] = stemmer.stem(tokenized_words[i])
    return tokenized_words


# In[11]:


def remove_stop_words_stemming(all_sen):
    
    clean_sen = []
    for sen in all_sen:
        #print(sen)
        tokenized_words = word_tokenize(sen)
        #print(tokenized_words)

        for word in tokenized_words:
            #print(word)
            if word in stopwords.words('english'):
                tokenized_words.remove(word)
        #tokenized_words = stemming(tokenized_words)
        clean_sen.append(tokenized_words)
    final_sen = []
    for i in range(len(clean_sen)):
        sen = ''
        #print(clean_sen[i])
        if clean_sen:
            for word in clean_sen[i]:
                sen += word +' '
        final_sen.append(sen) 
    return final_sen


# In[14]:


def clean_doc(sen_as_list):
    all_sen = lower_remove_words(sen_as_list)
    clean_data = remove_stop_words_stemming(all_sen)
    
    return clean_data
    
    


# In[17]:


#clean_doc(all_sen)


# In[ ]:




