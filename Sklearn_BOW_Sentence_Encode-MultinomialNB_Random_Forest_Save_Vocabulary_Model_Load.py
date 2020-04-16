#!/usr/bin/env python
# coding: utf-8

# In[9]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


# In[3]:


#https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk


# In[4]:


# Read Data
df = pd.read_csv('comp_pf_npf_data.csv')


# In[5]:


df.columns


# In[6]:


#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)


# In[7]:


text_counts= cv.fit(df['sen'])


# In[10]:


# Save the Vector
pickle.dump(cv, open('BOW_Vector/vectorizer.sav', 'wb'))


# In[11]:


#Print Vocabulary
text_counts.vocabulary_


# In[12]:


text_counts = cv.transform(df['sen'])


# In[ ]:


#text_counts.vocabulary_


# In[13]:


# MultinomialNB Model build on whole data
from sklearn.naive_bayes import MultinomialNB
clf_whole_data_MNB = MultinomialNB().fit(text_counts, df['label'])


# In[14]:


# Save the MultinomialNB Model on Complete Data
# pickling the model
pickle.dump(clf_whole_data_MNB, open('MultinomialNB_Model_BOW/MultinomialNB_model.sav', 'wb'))


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(text_counts, df['label'], test_size=0.3, random_state=1)


# In[13]:


# Build on Train dataset
from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


# In[22]:


confusion_matrix(y_test, predicted)


# In[27]:


tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()


# In[28]:


print(tn, fp, fn, tp)


# In[ ]:





# In[16]:


from sklearn.ensemble import RandomForestClassifier


# In[20]:


# Random Forest Build on Complete Data
clf_whole_data_RF=RandomForestClassifier(n_estimators=100)


# In[21]:


clf_whole_data_RF.fit(text_counts, df['label'])


# In[22]:


# Save the Random Forest Model build on Complete Data
# pickling the model
pickle.dump(clf_whole_data_RF, open('Random_Forest_Model_BOW/RF_BOW_model.sav', 'wb'))


# In[16]:


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)


# In[17]:


#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)


# In[30]:


y_pred=clf.predict(X_test)


# In[31]:


from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[32]:


from sklearn.metrics import confusion_matrix


# In[33]:


confusion_matrix(y_test, y_pred)


# In[36]:


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


# In[37]:


print(tn, fp, fn, tp)


# In[ ]:


# Load the Model and Vector
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
classifier = pickle.load(open('classifier.sav', 'rb'))


# In[ ]:


text = 'Do not purchase this product. My cell phone blast when I switched the charger'


# In[ ]:


text_vector = vectorizer.transform([text])
result = classifier.predict(text_vector)


# In[ ]:


result

