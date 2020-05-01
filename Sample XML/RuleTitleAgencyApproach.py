#----------------------------------------------Rule Title & Agency Approach---------------------------------------------
import pandas as pd
import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
porter = PorterStemmer()
from string import punctuation
from ast import literal_eval

# Keep only meaningful words in rule titles
es_rules=pd.read_excel('Data/TDM Studio/Sample XML/es_rules.xlsx')
print(es_rules.info())

title_words=[]
for text in es_rules['rule_title']:
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if not re.match('.*\d+', token)]    # remove tokens containing numbers
    tokens = [token for token in tokens if (token not in ('cy', 'fy')) & (len(token)>1)]
    tagged = nltk.pos_tag(tokens)
    text_clean = [x for (x, y) in tagged if y in ('FW','JJ','JJR','JJS','NN','NNS','NNP','NNPS',
                                                  'RB','RBR','RBS','RP','VB','VBD','VBG','VBN','VBP','VBZ')]
    title_words.append(text_clean)
es_rules['title_words']=title_words
es_rules.to_excel('Data/TDM Studio/Sample XML/rule_titles.xlsx',index=False)

# Function to remove multiple spaces
def remove_spaces(text):
    text=re.sub(' +',' ',text).strip()
    return text

# Function to match whole words (case sensitive)
def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w)).search

# Function to determine if a sentence contains 3+ tokens in a list
def sent_search_title(sent_stem,title):
    title_stem=[porter.stem(word.lower()) for word in title]
    rel=0
    match=len({w for w in sent_stem if w in title_stem})
    if len(title_stem)>=3:
        if match>=3:
            rel=1
    else:
        if match==len(title_stem):
            rel=1
    return rel

# Find rule title and agency name in text
df=pd.read_pickle('Data/TDM Studio/Sample XML/parsed_xml_BBD_clean.pkl')
es_rules=pd.read_excel('Data/TDM Studio/Sample XML/rule_titles.xlsx')
es_rules.loc[:,'title_words'] = es_rules.loc[:,'title_words'].apply(lambda x: literal_eval(x))
es_rules=es_rules.reset_index(drop=True)

has_title=[]
has_title_agency=[]
for text in df['Text']:
    text = text.replace('\n', ' ').replace('\r', '')
    text=remove_spaces(text)
    text_sentence=sent_tokenize(text)
    sent_rel = 0
    rel=0
    stop=False
    for sent in text_sentence:
        sent_token = word_tokenize(sent.lower())
        sent_token = [w for w in sent_token if w not in punctuation]
        sent_stem = [porter.stem(word) for word in sent_token]
        index=0
        for title in es_rules['title_words']:
            if len(title)>1:
                if sent_search_title(sent_stem,title)==1:
                    sent_rel=1
                    agency_name=es_rules['agency_name'][index]
                    dept_name=es_rules['department_name'][index]
                    agency_acronym=es_rules['agency_acronym'][index]
                    dept_acronym=es_rules['department_acronym'][index]
                    if (findWholeWord(agency_acronym)(text)!=None) | (findWholeWord(dept_acronym)(text)!=None) | \
                            (findWholeWord(agency_name)(text.lower()) != None) | (findWholeWord(dept_name)(text.lower())!=None):
                        rel=1
                        stop=True
                        break
            index = index + 1
        if stop:
            break
    has_title.append(sent_rel)
    has_title_agency.append(rel)
df['matchTitle']=has_title
df['matchTitleAgency']=has_title_agency
df.to_pickle('Data/TDM Studio/Sample XML/parsed_xml_clean_ruleTitle.pkl')
