"""

@author: daytonpaul@ MABA UMN
"""

import streamlit as st
import pandas as pd

#import plotly.express as px
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
from PIL import Image
import os
import spacy
from spacy.lang.en.examples import sentences
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
stopwords=list(STOP_WORDS)
from string import punctuation
punctuation=punctuation+ '\n'
from sentence_transformers import SentenceTransformer, util
import scipy.spatial
import pickle as pkl
embedder = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import torchvision
import re
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import heapq
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Rome! Your vacation Destination and Historic Wonderland')
st.markdown('Rome is a city filled with thousands of years of history.')
st.markdown('Finding the perfect hotel in the perfect spot can lead to a fulling vacation that will provide you with memories that last a life time.')
st.markdown('Search for your desired characteritics, and we will provide accommodations most suitable to you.')

st.image("https://media.timeout.com/images/105211701/image.jpg")

#@st.cache(persist=True)

dataset = st.container()
modelz = st.container()
features = st.container()


# Rome data importation and cleaning
with dataset:
    @st.cache(persist=True)
    def load_data():
        RomeReviewList = pd.read_csv('hotelReviewsInRome__en2019100120191005.csv')
        RomeReviewList['hotelName'] = RomeReviewList['hotelName'].str.split('\n').str[0]
        RomeReviewList['hotelName'] = RomeReviewList['hotelName'].str.split(' ').str[4:]
        RomeReviewList['hotelName'] = RomeReviewList['hotelName'].apply(lambda x: ' '.join(map(str, x)))
        return RomeReviewList

    RomeReviewList = load_data()

#    @st.cache(persist=True, allow_output_mutation=True)
#    def combine_review():
    RomeReviewCombined = RomeReviewList.sort_values(['hotelName']).groupby('hotelName', sort = False).review_body.apply(''.join).reset_index(name='all_review')
    #Combining reviews for encoding
    RomeReviewCombined['all_review'] = RomeReviewCombined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
#        return RomeReviewCombined

    #RomeReviewCombined = combine_review()

    @st.cache(persist=True)
    def lower_case(input_str):
        input_str = input_str.lower()
        return input_str

    RomeReviewCombined['all_review']= RomeReviewCombined['all_review'].apply(lambda x: lower_case(x))

    Rome_sentences = RomeReviewCombined.set_index("all_review")
    Rome_sentences = Rome_sentences["hotelName"].to_dict()
    Rome_sentences_list = list(Rome_sentences.keys())

    Rome = RomeReviewCombined

    Rome_sentences_list = [str(d) for d in tqdm(Rome_sentences_list)]

    corpus = Rome_sentences_list

    @st.cache(persist=True)
    def corp_embed():
        corpus_embeddings = embedder.encode(corpus,show_progress_bar=False)
        return corpus_embeddings
#    st.write(corpus_embeddings)

    corpus_embeddings = corp_embed()

with modelz:
    st.header('Search for Rome your way.')
#Get User Input
    queries = []
    query = st.text_input('Rome hotel lookup:')
    st.write('The current hotel query is:', query)


    embeddings = model.encode(corpus)
#        return embeddings

#    embeddings = model_embed()
#    st.write(embeddings)

#Generating Hotel Summaries to display
    hotel_summaries = pd.DataFrame(columns = ['Hotel', 'Summary'])
    stopword = set(STOPWORDS)
#    stopword = nltk.corpus.stopwords.words('english')

    Rome_summary = RomeReviewList.sort_values(['hotelName']).groupby('hotelName', sort = False).review_body.apply(''.join).reset_index(name='all_review')

    #word_frequencies = {}
    @st.cache(persist=True)
    def hotel_summ():
        Rome_summary = RomeReviewList.sort_values(['hotelName']).groupby('hotelName', sort = False).review_body.apply(''.join).reset_index(name='all_review')
        hotel_summaries = pd.DataFrame(columns = ['Hotel', 'Summary'])
        stopword = set(STOPWORDS)
        #nltk.corpus.stopwords.words('english')
        for hotel in range(len(Rome_summary)):
            word_frequencies = {}
            mini_corpus = Rome_summary.iloc[hotel,1]
            #  print(mini_corpus)
            for word in nltk.word_tokenize(mini_corpus):
                if word not in stopwords:
                  if word not in word_frequencies:
                    word_frequencies[word] = 1
                  else:
                    word_frequencies[word] += 1
            maximum_frequency = max(word_frequencies.values())
            for word in word_frequencies:
                word_frequencies[word] = (word_frequencies[word]/maximum_frequency)
            sentence_list = nltk.sent_tokenize(mini_corpus)
            #  print(sentence_list)
            sentence_scores = {}
            for sent in sentence_list:
                for word in nltk.word_tokenize(sent.lower()):
                    if word in word_frequencies:
                        if len(sent.split(' ')) < 30:
                            if sent not in sentence_scores.keys():
                                sentence_scores[sent] = word_frequencies[word]
                            else:
                                sentence_scores[sent] += word_frequencies[word]
            summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
            summary = ' '.join(summary_sentences)
            #  print(summary)
            hotel_summaries = hotel_summaries.append({'Hotel' : RomeReviewCombined.iloc[hotel,0], 'Summary': summary}, ignore_index = True)
        return hotel_summaries


    hotel_descriptions = hotel_summ()

#Adding and caching stopwords to increase speed
    @st.cache(persist=True)
    def stopword():
        stopwords = set(STOPWORDS)
        stopwords.add('room')
        stopwords.add('rooms')
        stopwords.add('hotel')
        stopwords.add('rome')
        return stopwords

    stopwords_cust = stopword()
#Output generation
with features:


#    def outputs(query):

    queries = re.split('[!?.]', query)
    queries = [i for i in queries if i]

    top_k = min(5, len(corpus))
    for query in queries:

        #query_embedding = model.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
        #cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        #top_results = torch.topk(cos_scores, k=top_k)

        st.write("\n\n======================\n\n")
        st.write("Hotels that best accomodate the request:", query)
        st.write("\nTop 5 most similar hotels to your request:")
'''
        for score, idx in zip(top_results[0], top_results[1]):
    #            st.write("(Score: {:.4f})".format(score))
    #            st.write(corpus[idx], "(Score: {:.4f})".format(score))
            row_dict = Rome.loc[Rome['all_review']== corpus[idx]]
            inter_frame = row_dict['hotelName'].to_frame().T
            inter_frame2 = np.asarray(inter_frame)
            st.subheader(inter_frame2[0,0] , "\n")
            if score > .45:
                st.markdown("_We show this as a great fit!_")
            elif score > .35 and score <= .45:
                 st.markdown("_We show this as a good fit!_")
            else:
                st.markdown('_We show this as a fair fit!_')
            st.write('This hotel is most frequently described as:')
            inter_summ = np.asarray(hotel_descriptions.loc[hotel_descriptions['Hotel'] == inter_frame2[0,0]].Summary)
            st.write(inter_summ[0])
            wordcloud = WordCloud(stopwords = stopwords_cust, max_words = 30, max_font_size=50,
                                    colormap='Set2',
                                    collocations=True, background_color="white").generate(corpus[idx])
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')

            st.pyplot()
            #st.plyplt.axis('off')
            #plt.show()
'''
