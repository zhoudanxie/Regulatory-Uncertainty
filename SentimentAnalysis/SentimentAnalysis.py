# Sentiment Analysis
# Reference: http://kaichen.work/?p=399

# Used the following four lexicons:
#- NLTK Vader
#- Harvard General Inquirer (GI) dictionary
#- Loughran and McDonald (2011) (LM) dictionary
#- Young and Soroka (2012) Lexicoder Sentiment Dictionary (LSD)

# Import packages
import pandas as pd
import re
from nltk.corpus import stopwords
import pickle
import numpy as np
import spacy
nlp=spacy.load('en_core_web_sm',disable=['parser','ner'])

# Load data
df=pd.read_pickle('news_RegUncertain.pkl')

# Negation list
negate = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt",
          "doesnt", "ain't", "aren't", "can't", "couldn't", "daren't", 
          "didn't", "doesn't", "dont", "hadnt", "hasnt", "havent", "isnt", 
          "mightnt", "mustnt","neither", "don't", "hadn't", "hasn't", 
          "haven't", "isn't", "mightn't", "mustn't", "neednt", "needn't",
          "never", "none", "nope", "nor", "not", "nothing", "nowhere", 
          "oughtnt", "shant", "shouldnt", "wasnt", "werent", "oughtn't",
          "shan't", "shouldn't", "wasn't", "weren't", "without", "wont",
          "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite",
          "no", "nobody"]

# LM sentiment word list
LMlist=pd.read_csv('LoughranMcDonald_SentimentList.csv')
LMneg=LMlist['Negative'].tolist()
LMpos=LMlist[LMlist['Positive'].notnull()]['Positive'].tolist()
lmdict={'Negative':[w.lower() for w in LMneg], 'Positive':[w.lower() for w in LMpos]}

# Negation function
def negated(word):
    #Determine if preceding word is a negation word
    if word.lower() in negate:
        return True
    else:
        return False

# Lemmatizer
def lemmatizer(text):
    doc=nlp(text)
    lemmas=[token.lemma_ for token in doc if not token.is_punct | token.is_space]
    return lemmas

# Function to count pos/neg words
def tone_count_with_negation_check(dict, article):
    """
    Count positive and negative words with negation check. Account for simple negation only for positive words.
    Simple negation is taken to be observations of one of negate words occurring within three words
    preceding a positive words.
    """
    pos_count = 0
    neg_count = 0
 
    pos_words = []
    neg_words = []
 
    input_words = lemmatizer(article)
 
    word_count = len(input_words)
 
    for i in range(0, word_count):
        if input_words[i] in dict['Negative']:
            neg_count += 1
            neg_words.append(input_words[i])
        if input_words[i] in dict['Positive']:
            if i >= 3:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]) or negated(input_words[i - 3]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 2:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 1:
                if negated(input_words[i - 1]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 0:
                pos_count += 1
                pos_words.append(input_words[i])
    
    '''
    print('The results with negation check:', end='\n\n')
    print('The # of positive words:', pos_count)
    print('The # of negative words:', neg_count)
    print('The list of found positive words:', pos_words)
    print('The list of found negative words:', neg_words)
    print('\n', end='')
    '''
    
    results = [word_count, pos_count, neg_count, pos_words, neg_words]
 
    return results

# Run through all articles
totalWordCount=[]
LMposCount=[]
LMnegCount=[]
LMposWords=[]
LMnegWords=[]
for text in df['Text']:
    results=tone_count_with_negation_check(lmdict, text)
    totalWordCount.append(results[0])
    LMposCount.append(results[1])
    LMnegCount.append(results[2])
    LMposWords.append(results[3])
    LMnegWords.append(results[4])

df['totalWordCount']=totalWordCount
df['LMposCount']=LMposCount
df['LMnegCount']=LMnegCount
df['LMposWords']=LMposWords
df['LMnegWords']=LMnegWords

# Harvard GI dictionary
with open('GIposWords.txt','rb') as fp:
    GIposWords=pickle.load(fp)
with open('GInegWords.txt','rb') as fp:
    GInegWords=pickle.load(fp)
    
gidict={'Negative':[w.lower() for w in GInegWords], 'Positive':[w.lower() for w in GIposWords]}

# Run through all articles
GIposCount=[]
GInegCount=[]
GIposWords=[]
GInegWords=[]
for text in df['Text']:
    results=tone_count_with_negation_check(gidict, text)
    GIposCount.append(results[1])
    GInegCount.append(results[2])
    GIposWords.append(results[3])
    GInegWords.append(results[4])

df['GIposCount']=GIposCount
df['GInegCount']=GInegCount
df['GIposWords']=GIposWords
df['GInegWords']=GInegWords

# Lexicoder Sentiment Dictionary
LSDlist=pd.read_csv('LSDsentimentWords.csv')
LSDneg=LSDlist[LSDlist['LSDnegative'].notnull()]['LSDnegative'].tolist()
LSDpos=LSDlist[LSDlist['LSDpositive'].notnull()]['LSDpositive'].tolist()
lsddict={'Negative':[w.lower() for w in LSDneg], 'Positive':[w.lower() for w in LSDpos]}

# Run through all articles
LSDposCount=[]
LSDnegCount=[]
LSDposWords=[]
LSDnegWords=[]
for text in df['Text']:
    results=tone_count_with_negation_check(lsddict, text)
    LSDposCount.append(results[1])
    LSDnegCount.append(results[2])
    LSDposWords.append(results[3])
    LSDnegWords.append(results[4])

df['LSDposCount']=LSDposCount
df['LSDnegCount']=LSDnegCount
df['LSDposWords']=LSDposWords
df['LSDnegWords']=LSDnegWords

# Calculate sentiment scores
df['LMscore']=(df['LMposCount']-df['LMposCount'])/df['totalWordCount']
df['GIscore']=(df['GIposCount']-df['GIposCount'])/df['totalWordCount']
df['LSDscore']=(df['LSDposCount']-df['LSDposCount'])/df['totalWordCount']

# NLTK Vader Sentiment
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
vader=SentimentIntensityAnalyzer()

# Apply to articles
scores=df['Text'].apply(vader.polarity_scores)
scores_df=pd.DataFrame.from_records(scores)

scores_df=scores_df.rename(columns={'neg':'VaderNeg','neu':'VaderNeu',
        'pos':'VaderPos','compound':'VaderScore'})
df=df.join(scores_df)

# LM Uncertainty Dictionary
LMuncertain=LMlist[LMlist['Uncertainty'].notnull()]['Uncertainty'].tolist()
uncertaindict={'Uncertainty': [w.lower() for w in LMuncertain]}

# Function to count uncertainty words
def uncertainty_count(dict, article):
    uncertain_count=0
    uncertain_words=[]
    
    input_words = lemmatizer(article)
 
    word_count = len(input_words)
 
    for i in range(0, word_count):
        if input_words[i] in dict['Uncertainty']:
            uncertain_count += 1
            uncertain_words.append(input_words[i])
    
    results=[uncertain_count, uncertain_words]
    return results
    
# Run through all articles
UncertaintyCount=[]
UncertaintyWords=[]
for text in df['Text']:
    results=uncertainty_count(uncertaindict, text)
    UncertaintyCount.append(results[0])
    UncertaintyWords.append(results[1])

df['UncertaintyCount']=UncertaintyCount
df['UncertaintyWords']=UncertaintyWords

# Save all results
df.to_pickle('AllSentiments.pkl')
