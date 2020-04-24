
#-------------------------------------------Search for BBD Keywords-----------------------------------------------------
# Import packages
import pandas as pd
import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
porter = PorterStemmer()
from string import punctuation

# Dataframe of news articles
df=pd.read_pickle('Data/TDM Studio/Sample XML/parsed_xml.pkl')

# BBD's three sets of keywords
uncertain=['uncertainty','uncertain']
econ=['economic','economy']
policy=['regulation', 'deficit','legislation', 'congress', 'white house', 'Federal Reserve', 'the Fed', 'regulations',
        'regulatory', 'deficits','congressional', 'legislative', 'legislature']
regulation=["banking supervision", "bank supervision", "glass-steagall", "tarp", "thrift supervision", "dodd-frank",
            "financial reform", "commodity futures trading commission", "cftc", "house financial services committee",
            "basel", "capital requirement", "Volcker rule", "bank stress test", "securities and exchange commission",
            "sec", "deposit insurance", "fdic", "fslic", "ots", "occ", "firrea", "truth in lending", "union rights",
            "card check", "collective bargaining law", "national labor relations board", "nlrb", "minimum wage",
            "living wage", "right to work", "closed shop", "wages and hours", "workers compensation",
            "advance notice requirement", "affirmative action", "at-will employment", "overtime requirements",
            "trade adjustment assistance", "davis-bacon", "equal employment opportunity", "eeo", "osha", "antitrust",
            "competition policy", "merger policy", "monopoly", "patent", "copyright", "federal trade commission",
            "ftc", "unfair business practice", "cartel", "competition law", "price fixing", "class action",
            "healthcare lawsuit", "tort reform", "tort policy", "punitive damages", "medical malpractice",
            "energy policy", "energy tax", "carbon tax", "cap and trade", "cap and tax", "drilling restrictions",
            "offshore drilling", "pollution controls", "environmental restrictions", "clean air act", "clean water act",
            "environmental protection agency", "epa", "immigration policy"]

# Function to Porter stem words in a list (stem every word)
def stem_list(words):
    for term in words:
        term_split=term.split()
        for word in term_split:
            stem=porter.stem(word)
            term_split=[w.replace(word, stem) for w in term_split]
        new_term=" ".join(term_split)
        words=[w.replace(term, new_term) for w in words]
    return words

# Stem BBD keywords list
uncertainStem=stem_list(uncertain)
econStem=stem_list(econ)
policyStem=stem_list(policy)
regulationStem=stem_list(regulation)

# Function to remove multiple spaces
def remove_spaces(text):
    text=re.sub(' +',' ',text).strip()
    return text

# Function to match whole words
def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

# Find keywords in text
def find_keyword(keywords):
    rel_list=[]
    for text in df['Text']:
        text = text.replace('\n', ' ').replace('\r', '')
        text=remove_spaces(text)
        text_token = word_tokenize(text)
        text_token = [w for w in text_token if w not in punctuation]
        text_stem=text
        for word in text_token:
            stem=porter.stem(word)
            text_stem=text_stem.replace(word,stem.lower())

        for term in keywords:
            rel=0
            if findWholeWord(term)(text_stem)!=None:
                rel=1
                match=term
                break
            else:
                match=''
        rel_list.append(rel)
        #print(rel,match)
    return rel_list

df['relUncertain']=find_keyword(uncertainStem)
df['relEcon']=find_keyword(econStem)
df['relPolicy']=find_keyword(policyStem)
df['relRegulation']=find_keyword(regulationStem)
df.to_pickle('Data/TDM Studio/Sample XML/parsed_xml_BBD.pkl')

#---------------------------------------------Aggregate Data-----------------------------------------------------------
import datetime

df=pd.read_pickle('Data/TDM Studio/Sample XML/parsed_xml_BBD.pkl')
print(df.info())

# Limit data to news only
df=df[df['Type']=='News'].reset_index(drop=True)

# Convert dates
df['StartDate']=df['StartDate'].astype('datetime64[ns]')
df['EndDate']=df['EndDate'].astype('datetime64[ns]')
df['Year']=df['StartDate'].astype('datetime64[ns]').dt.year
df['Month']=df['StartDate'].astype('datetime64[ns]').dt.month

# Clean duplicated news articles due to overlapped databases
df=df[((df['PubbTitle']=='The Washington Post') & (df['StartDate']>datetime.datetime(1996,12,3)))
                  | (df['PubbTitle']!='The Washington Post')]
df=df[((df['PubbTitle']=='Los Angeles Times') & (df['StartDate']>datetime.datetime(1996,12,3)))
                  | (df['PubbTitle']!='Los Angeles Times')]
df=df[((df['PubbTitle']=='Chicago Tribune') & (df['StartDate']>datetime.datetime(1996,12,3)))
                  | (df['PubbTitle']!='Chicago Tribune')]

df.loc[df['PubbTitle']=='New York Times','Newspaper']='New York Times'
df.loc[(df['PubbTitle']=='The Washington Post') | (df['PubbTitle']=='The Washington Post (pre-1997 Fulltext)'),
    'Newspaper']='The Washington Post'
df.loc[(df['PubbTitle']=='Los Angeles Times') | (df['PubbTitle']=='Los Angeles Times (pre-1997 Fulltext)'),
    'Newspaper']='Los Angeles Times'
df.loc[(df['PubbTitle']=='Chicago Tribune') | (df['PubbTitle']=='Chicago Tribune (pre-1997 Fulltext)'),
    'Newspaper']='Chicago Tribune'

# Articles with keywords in EU, PU, EPU, RPU, REPU
df.loc[(df['relUncertain']==1) & (df['relEcon']==1),'relEU']=1
df.loc[(df['relUncertain']==1) & (df['relPolicy']==1),'relPU']=1
df.loc[(df['relUncertain']==1) & (df['relEcon']==1) & (df['relPolicy']==1),'relEPU']=1
df.loc[(df['relUncertain']==1) & (df['relPolicy']==1) & (df['relRegulation']==1),'relRPU']=1
df.loc[(df['relUncertain']==1) & (df['relPolicy']==1) & (df['relEcon']==1) & (df['relRegulation']==1),'relREPU']=1

df.to_pickle('Data/TDM Studio/Sample XML/parsed_xml_BBD_clean.pkl')

# Generate daily count
dailyCount=df[['Newspaper','StartDate','relUncertain','relEcon','relPolicy','relRegulation','relEU',
               'relPU','relEPU','relRPU','relREPU']].groupby(['Newspaper','StartDate']).agg('sum').reset_index()
monthlyCount=df[['Newspaper','Year','Month','relUncertain','relEcon','relPolicy','relRegulation','relEU',
               'relPU','relEPU','relRPU','relREPU']].groupby(['Newspaper','Year','Month']).agg('sum').reset_index()
