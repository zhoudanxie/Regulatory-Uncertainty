#-------------------------------------------------Topic Modeling-------------------------------------------------------#
#---------------------------------------------------April 2020---------------------------------------------------------#

# References:
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/

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


#----------------------------------------Prepare Data for Topic Modeling------------------------------------------------
# Function to remove multiple spaces
def remove_spaces(text):
    text=re.sub(' +',' ',text).strip()
    return text

# Edit Porter stemmer to include "deregulat*"
prefixes={'de':''}
def stem_prefix(word, prefixes):
    original_word = word
    for prefix in sorted(prefixes, key=len, reverse=True):
        # Use subn to track the no. of substitution made.
        # Allow dash in between prefix and root.
        word, nsub = re.subn("{}[\-]?".format(prefix), "", word)
        if nsub > 0:
            return word
    return original_word

def porter_plus(word, prefixes=prefixes):
    return porter.stem(stem_prefix(word, prefixes))

# Function to Porter stem words in a list (stem every word)
def stem_list(words):
    for term in words:
        term_split=term.split()
        for word in term_split:
            stem=porter_plus(word.lower())
            term_split=[w.replace(word, stem) for w in term_split]
        new_term=" ".join(term_split)
        words=[w.replace(term, new_term) for w in words]
    return words

# Search terms
regStem=['regul','regulatori']
covidANDlist=['pneumonia','Wuhan']
covidORlist=['coronavirus','2019-nCoV', 'nCoV', 'COVID-19', 'COVID19','SARS-CoV-2', 'coronavirinae']

# Stem search terms
covidORstem=stem_list(covidORlist)
covidANDstem=stem_list(covidANDlist)
print(regStem,covidANDstem,covidORstem)

# Search keywords in text
df=pd.read_pickle('Data/TDM Studio/Sample XML/parsed_xml_BBD_clean.pkl')

relReg_list=[]
relCovid_list=[]
for text in df['Text']:
    text = text.replace('\n', ' ').replace('\r', '')
    text=remove_spaces(text)
    text_token = word_tokenize(text)
    text_token = [w for w in text_token if w not in punctuation]
    text_stem=[]
    for token in text_token:
        stem = porter_plus(token.lower())
        text_stem.append(stem)

    relReg=0
    relCovid=0
    matchReg=''
    matchCovid=''
    for term in regStem:
        if term in text_stem:
            relReg=1
            matchReg=term
            break

    for term in covidORstem:
        if term in text_stem:
            relCovid=1
            matchCovid=term
            break
    if relCovid==0:
        if (covidANDstem[0] in text_stem) & (covidANDstem[1] in text_stem):
            relCovid=1
            matchCovid=covidANDstem

    relReg_list.append(relReg)
    relCovid_list.append(relCovid)
    print(relReg,matchReg,relCovid,matchCovid)

df['relReg']=relReg_list
df['relCovid']=relCovid_list
df.loc[(df['relReg']==1) & (df['relCovid']==1),'regANDcovid']=1
df.loc[(df['relReg']==1) & (df['relCovid']==0),'regNOTcovid']=1


#---------------------------------------------Topic Modeling-----------------------------------------------------------
# Import more packages
import nltk; nltk.download('stopwords')
import numpy as np
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud, STOPWORDS

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'would','say','could','be','rule','new','year','go','also',
                   'chicago_tribune','washington_post','may','can','take','make',
                   'use','include','come','time','many','still','even','get','tell',
                   'percent','see','think','find','know','look','feel'])

# Import data
data = df[df['regORcovid']==1]['Text'].values.tolist()
IDs=df[df['regORcovid']==1]['ID'].values.tolist()
print(len(data))

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]
# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

# Tokenize words
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

# Create Bigram and Trigram Models
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[1]]])

# Remove Stopwords, Make Bigrams and Lemmatize
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
print(len(data_words_nostops))

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Form Trigrams
data_words_trigrams=make_trigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Remove stopwords once again after lemmatization
data_lemmatized=remove_stopwords(data_lemmatized)

print(data_lemmatized[:1])
print(len(data_lemmatized))

# Wordcloud of all texts
# Join the texts together
long_string=''
for text in data_lemmatized:
    string=', '.join(v for v in text)
    long_string=long_string+', '+string

# Creat a Wordcloud object
wordcloud=WordCloud(background_color="white", max_words=5000, contour_width=3,contour_color='steelblue',
                    width=500, height=300)

# Generate a word cloud
wordcloud.generate(long_string)
wordcloud.to_image()
wordcloud.to_file('Data/TDM Studio/Sample XML/WordcloudAll.jpg')

# Create the Dictionary and Corpus needed for Topic Modeling
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized
print(len(texts))

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])
print(len(corpus))

# See what word a given id corresponds to, pass the id as a key to the dictionary
print(id2word[1])

# Human readable format of corpus (term-frequency)
print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=9,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
# Print the Keyword in the topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Model Perplexity and Coherence Score
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Find the optimal number of topics for LDA
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=16, step=1)

# Show graph
limit=16; start=2; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 2))

# Select the model and print the topics
optimal_model = model_list[5]
topicNo=9   # Number of topics in the optimal model
model_topics = optimal_model.show_topics(formatted=False,num_topics=topicNo)
pprint(optimal_model.print_topics(num_words=10))

# Visualize the topics-keywords
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(optimal_model, corpus, id2word, sort_topics=False)
pyLDAvis.show(vis)
pyLDAvis.save_html(vis,'Data/TDM Studio/Sample XML/TopicVisual.html')

# Finding the dominant topic in each sentence
def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

# Format
df_dominant_topic = df_topic_sents_keywords
df_dominant_topic.columns=['Dominant_Topic','Perc_Contribution','Topic_Keywords','Text']
df_dominant_topic['ID']=IDs
df_dominant_topic.to_excel(r'C:\Users\zxie\Box Sync\GWRSC\USDA 3rd Coop\Data\Programming\USDA\USDA_Text_DominantTopic.xlsx',index=False)

# The most representative sentence for each topic
# Display setting to show more characters in column
pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
                                            axis=0)

# Reset Index
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Show
sent_topics_sorteddf_mallet.head(10)

# Word Clouds of Top N Keywords in Each Topic
# 1. Wordcloud of Top N words in each topic
#cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
cols=['#033C5A','AA9868','#0190DB','#FFC72C','#A75523','#008364','#78BE20','#C9102F',
      '#033C5A','AA9868','#0190DB','#FFC72C','#A75523','#008364','#78BE20','#C9102F']   # GW colors

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=500,
                  height=300,
                  max_words=20,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = optimal_model.show_topics(formatted=False,num_topics=topicNo)

fig, axes = plt.subplots(9, 1, figsize=(40,40), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(topics[i][0]+1), fontdict=dict(size=20))
    plt.gca().axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
plt.savefig('Data/TDM Studio/Sample XML/Wordcloud_byTopic.jpg')

# Topic distribution
# Sentence Coloring of N Sentences
def topics_per_document(model, corpus):
    corpus_sel = corpus
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)

dominant_topics, topic_percentages = topics_per_document(model=optimal_model, corpus=corpus)

# Distribution of Dominant Topics in Each Document
df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

# Total Topic Distribution by actual weight
topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

# Top 3 Keywords for each Topic
topic_top3words = [(i, topic) for i, topics in optimal_model.show_topics(formatted=False,num_topics=topicNo)
                                 for j, (topic, wt) in enumerate(topics) if j < 3]

df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
df_top3words.reset_index(level=0,inplace=True)

# Save topic weights for each doc
df_topic_weight=topic_weightage_by_doc
df_topic_weight['ID']=IDs
print(df_topic_weight.info())
df_topic_weight.to_csv('Data/TDM Studio/Sample XML/TopicWeight.csv',index=False)

# Plot
from matplotlib.ticker import FuncFormatter

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 25), dpi=120, sharey=True)

# Topic Distribution by Dominant Topics
ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='#AA9868')
ax1.set_xticks(range(df_top3words.topic_id.unique().__len__()))
tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x+1)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
ax1.xaxis.set_major_formatter(tick_formatter)
ax1.tick_params(labelsize=20)
ax1.set_title('Number of News Articles by Dominant Topic', fontdict=dict(size=25))
ax1.set_ylabel('Number of News Articles',fontsize=20)
ax1.set_ylim(0, 250)

# Topic Distribution by Topic Weights
ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='#033C5A')
ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
ax2.xaxis.set_major_formatter(tick_formatter)
ax2.tick_params(labelsize=20)
ax2.set_title('Number of News Articles by Relevant Topic', fontdict=dict(size=25))
ax2.set_ylabel('Weighted Number of News Articles',fontsize=20)

plt.subplots_adjust(wspace=0,hspace=0.3)
plt.show()
plt.savefig('Data/TDM Studio/Sample XML/TopicDistribution.jpg')