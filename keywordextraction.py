#!/usr/bin/env python
"""
Contains functions for keyword extraction using a classifer trained on the Crowd500 dataset [Marujo2012]
"""

import os
import re
import random
import numpy as np
import pickle

import string
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stoplist = stopwords.words('english')

from gensim import corpora, models, similarities
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

from features import *


##################################################################
# functions to get train/test set and extract features from text #
##################################################################
def to_tfidf(documents):
  """
  Returns documents transformed to tf-idf vector space
  """
  texts = [[remove_punctuation(word) for word in document.lower().split() if word not in stoplist]
    for document in documents]
  dictionary = corpora.Dictionary(texts)
  corpus = [dictionary.doc2bow(text) for text in texts]
  tfidf = models.TfidfModel(corpus,normalize=True)
  corpus_tfidf = tfidf[corpus]    

  return {'dictionary':dictionary, 'corpus':corpus_tfidf, 'tfidf_model': tfidf}


def get_features_labels(data,corpus,dictionary,verbose):
  """
  Returns matrices X containing features and Y containing labels.
  Labels are 0 (not a keyword) and 1 (keyword).  
  """
  num_docs = len(data['documents'])
  
  for doc_idx in xrange(num_docs):
    text = data['documents'][doc_idx]
    keywords = data['keywords'][doc_idx]
    corpus_entry = corpus[doc_idx]
 
    # as keyword classification operates at the level of single word,
    # we define any word that occurs in a keyword phrase as a keyword
    separate_keywords = []
    for k in keywords: 
      separate_keywords.extend(remove_punctuation(k.lower()).split())

    # collect positive (keyword) and negative (non-keyword) examples
    positive_examples = separate_keywords
    num_positive = len(positive_examples)

    all_words = [remove_punctuation(w) for w in text.lower().split()]
    negative_examples = [w for w in all_words if (w not in positive_examples) and (w not in stoplist)]
    if len(negative_examples)>num_positive:
      negative_examples = random.sample(negative_examples,num_positive)
    num_negative = len(negative_examples)

    # balance the number of positive and negative examples
    if num_positive < num_negative:
      candidate_keywords = positive_examples + random.sample(negative_examples,num_positive)
      labels = np.array([1]*num_positive + [0]*num_positive)
    elif num_positive > num_negative:
      candidate_keywords = random.sample(positive_examples,num_negative) + negative_examples
      labels = np.array([1]*num_negative + [0]*num_negative)
    else:
      candidate_keywords = positive_examples + negative_examples
      labels = np.array([1]*num_positive + [0]*num_negative)

    # assemble labels
    if doc_idx==0:
      all_labels = labels
    else:
      all_labels = np.concatenate((all_labels,labels))

    # assemble features
    feature_set = extract_features(text,candidate_keywords,corpus_entry,dictionary)
    if doc_idx==0:
      all_features = feature_set['features']
    else:
      all_features = np.vstack((all_features,feature_set['features']))

    if verbose:
      print 'get_features_labels: extracted %d samples from document %d of %d' % (len(labels),doc_idx+1,num_docs)
  
  return {'features':all_features, 'labels':all_labels}


###########################################
# functions to perform keyword extraction #
###########################################

def get_keywordclassifier(preload):
  """
  Returns a keyword classifier trained and tested on dataset derived from Crowd500 [Marujo2012]
  """  
  classifier_type = 'logistic'
  if preload==1:
    train_XY = pickle.load(open('saved/trainXY_crowd500.pkl','rb'))
    test_XY = pickle.load(open('saved/testXY_crowd500.pkl','rb'))    
    model = pickle.load(open('saved/logisticregression_crowd500.pkl','rb'))
    
  else:
    # get training data from crowd500 corpus
    traindata = get_crowdd500_data('train')
    tx_traindata = to_tfidf(traindata['documents'])
    train_XY = get_features_labels(traindata,tx_traindata['corpus'],tx_traindata['dictionary'],1)
    pickle.dump(train_XY, open('saved/trainXY_crowd500.pkl','wb'))    
  
    # get test data from crowd500 corpus

    # use tf-idf dictionary learned on training data to transform test data
    dictionary = tx_traindata['dictionary']
    tfidf = tx_traindata['tfidf_model']
    texts = [[remove_punctuation(word) for word in document.lower().split() if word not in stoplist]
              for document in testdata['documents']]
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpus_tfidf = tfidf[corpus]    
    tx_testdata = {'dictionary':dictionary, 'corpus':corpus_tfidf, 'tfidf_model': tfidf}

    test_XY = get_features_labels(testdata,tx_testdata['corpus'],tx_testdata['dictionary'],1)
    pickle.dump(test_XY, open('saved/testXY_crowd500.pkl','wb')) 
    # train model for keyword classification 
    if classifier_type=='logistic':
      model = LogisticRegression()
      model = model.fit(train_XY['features'],train_XY['labels'])
      pickle.dump(model, open('saved/logisticregression_crowd500.pkl','wb'))    
    else:
      model = RandomForestClassifier(n_estimators=10)
      model = model.fit(train_XY['features'],train_XY['labels'])
      pickle.dump(model, open('saved/randomforest_crowd500.pkl','wb'))    

  # show performance of classifier
  in_sample_acc = cross_val_score(model, train_XY['features'], train_XY['labels'], cv=4)
  out_sample_acc = cross_val_score(model, test_XY['features'], test_XY['labels'], cv=4)
 
  return {'model': model, 'train_XY':train_XY, 'test_XY':test_XY}


def generate_candidates(text):
  """
  Returns candidate words that occur in named entities, noun phrases, or top trigrams
  """
  num_trigrams = 5
  named_entities = get_namedentities(text)
  noun_phrases = get_nounphrases(text)
  top_trigrams = get_trigrams(text,num_trigrams)

  return list(set.union(set(named_entities),set(noun_phrases),set(top_trigrams)))


def extract_keywords(text,keyword_classifier,top_k,preload):
  """
  Returns top k keywords using specified keyword classifier
  """
  # pre-processing to enable tf-idf representation
  if preload==1:
    preprocessing = pickle.load(open('saved/tfidf_preprocessing.pkl','rb'))
    dictionary = preprocessing['dictionary']
    tfidf = preprocessing['tfidf_model'] 


  text_processed = [remove_punctuation(word) for word in text.lower().split() if word not in stoplist]
  corpus = [dictionary.doc2bow(text_processed)]
  corpus_entry = tfidf[corpus][0]    

  # generate canddiate keywords
  candidate_keywords = generate_candidates(text)
  if len(candidate_keywords) < top_k:
    candidate_keywords = text_processed   

  # select from candidate keywords 
  feature_set = extract_features(text,candidate_keywords,corpus_entry,dictionary)
  predicted_prob = keyword_classifier.predict_proba(feature_set['features'])
  this_column = np.where(keyword_classifier.classes_==1)[0][0]
  sorted_indices = [i[0] for i in sorted(enumerate(predicted_prob[:,this_column]),key = lambda x:x[1],reverse = True)]
  chosen_keywords = [candidate_keywords[j] for j in sorted_indices[:top_k]]    
  
  return chosen_keywords
