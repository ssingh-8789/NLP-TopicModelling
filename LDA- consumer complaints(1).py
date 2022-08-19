

#author name - shivani singh
#author email - satyshi2015@gmail.com


import pandas as pd# to load the data
from sklearn.feature_extraction.text import TfidfVectorizer #convert text data into machine readable , CountVectorizer
from sklearn import decomposition#Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
#The input data is centered but not scaled for each feature before applying the SVD.
import matplotlib.pyplot as plt# for visulaization
import numpy as np
import re #regular expression for cleaning the data
import nltk #for preprocessing pipeline
from nltk.stem.porter import PorterStemmer#An algorithm for suffix stripping
from sklearn.model_selection import train_test_split#Split arrays or matrices into random train and test subsets.





df = pd.read_csv('consumer_compliants.csv')





df





nltk.download('punkt') # punkt library downloading from nltk , it is sentence tokenizer




df['Product'].value_counts()#Return a Series containing counts of unique rows in the DataFrame.





df['Company'].value_counts()#Return a Series containing counts of unique rows in the DataFrame.





complaints_df=df[['Consumer complaint narrative','Product','Company']].rename(columns={'Consumer complaint narrative':'complaints'})





pd.set_option('display.max_colwidth', -1)#Pandas have an options system to set the value of a specified option.
#using set_option() and Changing the number of columns to be displayed using display.max_columns.
complaints_df





X_train, X_hold = train_test_split(complaints_df, test_size=0.6, random_state=111)# spliting the data





X_train['Product'].value_counts()#Return a Series containing counts of unique rows in the DataFrame.





stemmer = PorterStemmer()#Create a new Porter stemmer.
#The main applications of Porter Stemmer include data mining and Information retrieval. 
#However, its applications are only limited to English words. Also,
#the group of stems is mapped on to the same stem and the output stem is not necessarily a meaningful word.


# now, below i am creating a function to tokenize the data using nltk word tokenizer, and filtering out the words that is less than 3 characters ,and sensitivie information that is masked as xxx.. in dataset, we will strip out if xx>2 else ignore that characters.




def tokenize(text):
    tokens = [word for word in nltk.word_tokenize(text) if (len(word) > 3 and len(word.strip('Xx/')) > 2) ] 
    #stems = [stemmer.stem(item) for item in tokens]
    return tokens


# now below i m initializing the tfidf vector, to take text data and convert it into numeric representation, into a vector format. will use lda algo, and lda needs only the count of the particular word , it does not need a normalized word.so we can either use count vectorizer or tfidf vectorizer keeping idf=false,and norm=none.by default tf use l2 norm, so the output data is normalized. but as norm is none it will behave as count vectorizer.
# 
# and stopword is english means use the top word, and using tokenixer, if you don't use tokenizer tfidf as inbuilt tokenizer to execute but since i have to filter out some word, we created our own tokenizer and calling it.
# 
# i have given max_document_frequency as 0.75 , when picking the words i am giving max features as 10,000. 
# so when picking the words make sure at least 75% document this word has, and if it goes above it means it is very common words, so ignore those words.
# 
# similarly that word has being 50 documents, if it is less than 50 documents it can be a rare word so filter out those words.
# 
# then passing complaints column in the training data frame to the fit transform method of vectorizer , it will give output as word vector, more like fature or numeric vector.
# 
# so in o/p you can see wherever the word it is 1, else 0.it is a sparse vector,it will have 10,000 elements in each row of this matrix.




vectorizer_tf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_df=0.75, min_df=50, max_features=10000, use_idf=False, norm=None)
tf_vectors = vectorizer_tf.fit_transform(X_train.complaints)





tf_vectors.A


# the features of this vector you will get from this get_feature_names().it will tell all the top 10,000 features that you asked, 




vectorizer_tf.get_feature_names()


# now the tf , that i have created we will pass it into LDA model.
# 
# LDA is a statistical model that allows a set of observation to be explain, by a group or unobserved group.
# 
# one typical application of LDA is basically topic modelling. which automatically classify documents and estimate the relevance to the documents.
# 
# so here in lda, i am passing components as 6, means we want to distribute doc in 6 topics.as here we have 6 products so gave 6 topics. but topic selection depends on domain knowledge that you are running it against.
# 
# here, running for 3 iteration, but you can choose more as running for more iteration can give better result.
# 
# the learning method is online, there are two method one is batch,batch uses all the data so,in every iteration what it's going to do is to replace the previous topic with the newly created topic vector, but online method is more preferrable method for large dataset.
# 
# the learning method returns mini batch of our dataframe, and the learning offset is basially telling you how do you want to wait the early iteration in this online learning process.
# 
# so online learning does incremental selection of topics.
# 
# n_jobs is telling that you solve the processor.
# 
# and random experiment is to reproduce my experiment.
# 
# after lda, i am calling the fit_transfer with the tf vector that i have created .
# and then i m printing my lda components. which ic nothing but my topics.




lda = decomposition.LatentDirichletAllocation(n_components=6, max_iter=3, learning_method='online', learning_offset=50, n_jobs=-1, random_state=111)

W1 = lda.fit_transform(tf_vectors)
H1 = lda.components_


# In[ ]:


w1 is containing output tranformed vector, it is going to score each document in each topic.
and h1 is going to have each components to it.





W1


# now below what i am doing is, i m selecting top word of each topic,so what are the top words that are contributing to each topic so that we can use the topwords to identify , what is the customer talking about in each and every topic. 
# 
# so i m selecting the top 15 words , and getting all the vectorizer feature name and assigning it to vocab, 
# and using a lambda function , which will basically take the vocab object and bring the topwords.
# and the taking topwords vector and passing it through H1 , and h1 is nothng but 6 topics that we have taken,
# it is going to iterate across, and then finally its going to print the topics .
# it is going to join all topic words ,
# the o/p will be for each topics what are the words that are contributing to it.


num_words=15

vocab = np.array(vectorizer_tf.get_feature_names())

top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_words-1:-1]]
topic_words = ([top_words(t) for t in H1])
topics = [' '.join(t) for t in topic_words]





topics


# now i am going to take this topic as input data against each row, what is prominent topic i m going to print it, just to see what is each and every input element what word is conributing,what topic it is.
# 
# so the colnames is all the 6 topics, in the row i have doc ids, so its going to take train dataframe,and is going to iterate and will create doc 0 ,1,2...
# 
# so it ois creating a document topic matrix.
# then i m creating pandas dataframe, taking input 6 weight vector, and using argmax picking the most prominent topics that document can belong, and last i m adding one more column called dominant topic.




colnames = ["Topic" + str(i) for i in range(lda.n_components)]
docnames = ["Doc" + str(i) for i in range(len(X_train.complaints))]
df_doc_topic = pd.DataFrame(np.round(W1, 2), columns=colnames, index=docnames)
significant_topic = np.argmax(df_doc_topic.values, axis=1)
df_doc_topic['dominant_topic'] = significant_topic






df_doc_topic





X_train.head()


# now i m going to take my hold on dataset, now i dont have model, i have done it at trainging, now i have new data that will be coming in on a daily basis , i want to classify it, so i have the lda model and i m calling the lda model transform method , i m passing the vectorizer , again the input hold.dataset also need to apply my vectorizer, bcz that is part of my pipeline, calling the transform and passing my complex dataset.
# just to run quickly given 5,




WHold = lda.transform(vectorizer_tf.transform(X_hold.complaints[:5]))


# below i m running the same method that i used above, to create a document topic matrix.only we will change the training dataset to hold dataset, and instead of w1 , passing w old vector. 




colnames = ["Topic" + str(i) for i in range(lda.n_components)]
docnames = ["Doc" + str(i) for i in range(len(X_hold.complaints[:5]))]
df_doc_topic = pd.DataFrame(np.round(WHold, 2), columns=colnames, index=docnames)
significant_topic = np.argmax(df_doc_topic.values, axis=1)
df_doc_topic['dominant_topic'] = significant_topic





df_doc_topic





X_hold.head()







