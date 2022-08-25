#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
 


# In[4]:


documents = pd.read_csv('news-data.csv', error_bad_lines=False)
documents.head()


# 1. The Components Matrix represents topics
# 2. The Features Matrix combines topics into documents
# 
# The first step is to create the TF-IDF matrix as follows.

# In[5]:


# use tfidf by removing tokens that don't appear in at least 50 documents
vect = TfidfVectorizer(min_df=50, stop_words='english')
 
# Fit and transform
X = vect.fit_transform(documents.headline_text)


# we will build the NMF model which will generate the Feature and the Component matrices.
# 

# In[6]:


# Create an NMF instance: model
# the 10 components will be the topics
model = NMF(n_components=10, random_state=5)
 
# Fit the model to TF-IDF
model.fit(X)
 
# Transform the TF-IDF: nmf_features
nmf_features = model.transform(X)



# It important to check the dimensions of the 3 tables:

# TF-IDF Dimensions:

# In[7]:


X.shape


# Features Dimensions:

# In[8]:


nmf_features.shape


# Components Dimensions:

# In[9]:


model.components_.shape


# We should add the column names to the Components matrix since these are the tokens (words) from the TF-IDF.
# based on our tokenizer we included any token of two or more characters that is why you will see some numbers. We could have also removed them. Both approaches are correct. 

# In[10]:


# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=vect.get_feature_names())
components_df


# We have created the 10 topics using NMF. Let’s have a look at the 10 more important words for each topic.

# In[11]:


for topic in range(components_df.shape[0]):
    tmp = components_df.iloc[topic]
    print(f'For topic {topic+1} the words with the highest value are:')
    print(tmp.nlargest(10))
    print('\n')


# 1.As we can see the topics appear to be meaningful. For example, Topic 3 seems to be about missing persons and investigations (police, probe, investigation, missing, search, seek etc)
# 
# 2.Since we defined the topics, we will show how you can get the topic of each document. Let’s say that we want to get the topic of the 55th document: ‘funds to help restore cossack’
# 
# 

# In[12]:


my_document = documents.headline_text[55]
my_document


# We will need to work with the Features matrix. So let’s get the 55th row:

# In[13]:


pd.DataFrame(nmf_features).loc[55]


# We look for the Topic with the maximum value which is the one of index 9 which is the 10th in our case (note that we started from 1 instead of 0). If we see the most important words of Topic 10 we will see that it contains the “funding“!
# 
# Note that if we wanted to get the index in once, we could have typed:

# In[14]:


pd.DataFrame(nmf_features).loc[55].idxmax()



# Finally, if we want to see the number of documents for each topic we can easily get it by typing:

# In[15]:


pd.DataFrame(nmf_features).idxmax(axis=1).value_counts()


# Let’s say that we want to assign a topic of a new unseen document. Then, we will need to take the document, to transform the TF-IDF model and finally to transform the NMF model. Let’s take the actual new head title from ABC news.

# In[17]:


my_news = """NY's Sean Patrick Maloney wins primary over progressive challenger after moving districts """
 
# Transform the TF-IDF
X = vect.transform([my_news])
# Transform the TF-IDF: nmf_features
nmf_features = model.transform(X)
 
pd.DataFrame(nmf_features)


# And if we want to get the index of the topic with the highest score:

# In[18]:


pd.DataFrame(nmf_features).idxmax(axis=1)


# As expected, this document was classified as Topic 10 (with index 9).

# In[ ]:




