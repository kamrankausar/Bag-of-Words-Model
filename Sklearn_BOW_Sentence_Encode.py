#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
import pandas as pd


# In[2]:


# Read Data
df = pd.read_csv('pf_npf.csv')


# In[4]:


data = df[:10]


# In[6]:


data


# In[17]:


#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)


# In[ ]:





# In[25]:


text_counts= cv.fit(data['data'])


# In[26]:


#Print Vocabulary
text_counts.vocabulary_


# In[27]:


text_counts = cv.transform(data['data'])


# In[29]:


#text_counts.vocabulary_


# In[ ]:




