# Customize stopwords with lemmatization

# Import packages
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pickle
import numpy as np
import spacy
nlp=spacy.load('en_core_web_sm',disable=['parser','ner'])

# Load data
df=pd.read_pickle('news_TopicModeling.pkl')

# Preprocess text
def preprocess_text(text):
    text=text.lower()   #lowercase
    text=re.sub("<!--?.*?-->","",text)  #remove tags
    text=re.sub("(\\d|\\W)+"," ",text)  #remove special charaters and digits
    return text

df['Text']=df['Text'].apply(lambda x:preprocess_text(x))
docs=df['Text'].tolist()

# Pre-defined stopwords
pre_stopwords=stopwords.words('english')

# Customize preprocessor for vectorizer
def my_preprocessor(text)
    doc=nlp(text)
    lemmas=[token.lemma_ for token in doc if not token.is_punct | token.is_space]
    lemmas_nostops=[w for w in lemmas if w not in pre_stopwords]
    texts_out=" ".join(lemmas_nostops)
    return texts_out
    
# Count term frequency across all documents
# Create a vocabulary of words, use customized preprocessor
cv=CountVectorizer(docs,preprocessor=my_preprocessor)
word_count_vector=cv.fit_transform(docs)
print(word_count_vector.shape)

# Show some words
print(list(cv.vocabulary_.keys())[:10])

# Use document counts to define stopwords
# Stopwords as words in <2 docs and >30% of docs
max=0.3
cv2=CountVectorizer(docs,preprocessor=my_preprocessor,min_df=2,max_df=max)
word_count_vector2=cv2.fit_transform(docs)
stopwords_wc=list(cv2.stop_words_)
print(len(stopwords_wc))

# Save stopwords
with open('stopwords_wc.txt','wb') as fp:
    pickle.dump(stopwords_wc,fp)

# Compute IDF
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

# Print IDF values
df_idf=pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=['idf_weights'])
# Sort ascending
df_idf=df_idf.sort_values('idf_weights').reset_index()
print(df_idf.head(10))

# Define stopwords as words with highest IDF (equivalent to words in only one doc) and words within the lowest X% of IDF
X=0.5
stopwords_idf=df_idf[(df_idf['idf_weights']==max(df_idf['idf_weights'])) | (df_idf['idf_weights']<df_idf['idf_weights'].quantile(X/100))]['index'].tolist()

# Save stopwords
with open('stopwords_idf.txt','wb') as fp:
    pickle.dump(stopwords_idf,fp)
