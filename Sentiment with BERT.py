#!/usr/bin/env python
# coding: utf-8

# In[1]:


# MPS acceleration is available on MacOS 12.3+
get_ipython().system('pip install torch torchvision torchaudio')


# In[2]:


get_ipython().system('pip install transformers requests beautifulsoup4 pandas numpy')


# In[5]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re 


# In[7]:


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# In[9]:


tokens = tokenizer.encode('I hated this , absolutely the worst',return_tensors='pt')


# In[11]:


result = model(tokens)


# In[13]:


result.logits


# In[14]:


int(torch.argmax(result.logits))+1


# In[16]:


r = requests.get('https://www.yelp.com/biz/mejico-sydney-2')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class':regex})
reviews = [result.text for result in results]


# In[19]:


reviews[0]


# In[20]:


import pandas as pd
import numpy as np


# In[21]:


df = pd.DataFrame(np.array(reviews), columns=['review'])


# In[22]:


df.head()


# In[24]:


def sentiment_score(review):
    tokens = tokenizer.encode(review,return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


# In[28]:


sentiment_score(df['review'].iloc[0])


# In[29]:


df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))


# In[30]:


df['review']


# In[31]:


df


# In[ ]:




