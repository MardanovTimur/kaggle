#!/usr/bin/env python
# coding: utf-8

# In[2]:
import numpy as np
import pandas as pd
import keras
import re
import tensorflow as tf
from matplotlib import pyplot as plt

from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from pymystem3 import Mystem
from sklearn.model_selection import train_test_split
stemmer = Mystem()


# In[3]:


filename = 'data/lenta-ru-news.csv'


# In[4]:


df = pd.read_csv(filename)


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


len(df.index)


# In[8]:


def year_extraction(row):
    return int(row['date'][0:4])


# In[9]:


df['year'] = df.apply(lambda row: year_extraction(row), axis=1)


# In[10]:


df.columns


# In[11]:


df.head()


# In[12]:


year_count_summary = df.groupby(['year']).size()
year_count_summary


# In[13]:

# новостей больше в определенные года по причинам


# In[14]:


def year_condition(row):
    return row['year'] > 1999


# In[15]:


df = df[df['year'] > 1999]


# In[17]:


#  plt.rcdefaults()
#  fig, ax = plt.subplots()

#  fig.figsize = (5, 20)
#  ax.barh(tags_count_chart.index, tags_count_chart, align='center')
#  ax.set_yticks(tags_count_chart.index)
#  ax.invert_yaxis()
#  ax.set_xlabel('Count of articles')
#  ax.set_title('Count of articles by tags (without Все)')
#  plt.show()


# In[18]:


#  print('Count of unique tags is: ', len(tags_count_chart.index))


# In[19]:


#  with pd.option_context('display.max_rows', -1, 'display.max_columns', 5):
#      print(tags_count_chart)


# In[20]:


df[df['tags'] == 'Все'].groupby(['year']).size()
# Категория Все - когда не можем однозначно отнести новость к этому классу.


# In[21]:


print('Топики - ', df.groupby('topic').size())


# In[22]:


df[df['topic'] == 'Крым'].groupby(['year']).size()
# новости про крым в определенные года.


# In[23]:


import pickle

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 1254
# This is fixed.
EMBEDDING_DIM = 100


# In[109]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[0-9a-z#+_]')
STOPWORDS = set(stopwords.words('russian'))

stemmer = Mystem()

def clean_text(text):
    try:
        text = text.lower() # lowercase text
    except:
        text = str(text).lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = text.replace('x', '')
    text = " ".join(stemmer.lemmatize(word)[0] for word in text.split() if word not in STOPWORDS)
    return text


# In[ ]:


stemmer = Mystem()
def read_text_dataframe_generator():
    x = []
    for row in df['text']:
        x.append(clean_text(row))
    return x

X = read_text_dataframe_generator()


# In[ ]:


#  tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)


#  # In[ ]:


#  tokenizer.fit_on_texts(X)


# In[ ]:


with open('data/x_text_dumped_array.dump', 'wb') as file:
    pickle.dump(X, file)


# In[ ]:




