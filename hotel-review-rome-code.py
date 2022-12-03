"""

@author: daytonpaul@ MABA UMN
"""

import streamlit as st
import pandas as pd

import plotly.express as px
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
from PIL import Image
import os
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
stopwords=list(STOP_WORDS)
from string import punctuation
punctuation=punctuation+ '\n'
from sentence_transformers import SentenceTransformer
import scipy.spatial
import pickle as pkl
embedder = SentenceTransformer('all-MiniLM-L6-v2')
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import torchvision

st.title('Rome! Your vacation Destination and Historic Wonderland')
st.markdown('Rome is a city filled with thousands of years of history.')
st.markdown('Finding the perfect hotel in the perfect spot can lead to a fulling vacation that will provide you with memories that last a life time.')
st.markdown('Search for your most important characteristics for you, and we will provide accommodations most suitable to your desires.')

st.image("https://media.timeout.com/images/105211701/image.jpg")

@st.cache(persist=True)

# Rome data importation and cleaning
RomeReviewList = pd.read_csv('hotelReviewsInRome__en2019100120191005.csv')
RomeReviewList['hotelName'] = RomeReviewList['hotelName'].str.split('\n').str[0]
RomeReviewList['hotelName'] = RomeReviewList['hotelName'].str.split(' ').str[4:]
RomeReviewList['hotelName'] = RomeReviewList['hotelName'].apply(lambda x: ' '.join(map(str, x)))
RomeReviewCombined = RomeReviewList.sort_values(['hotelName']).groupby('hotelName', sort = False).review_body.apply(''.join).reset_index(name='all_review')

#Combining reviews for encoding
import re

RomeReviewCombined['all_review'] = RomeReviewCombined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

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
corpus_embeddings = embedder.encode(corpus,show_progress_bar=False)

#Get User Input
queries = []

query = st.text_input('Rome hotel lookup', 'E.g., near the Vatican')
st.write('The current hotel query is:', queries)

queries = re.split('[!?.]', query)
queries = [i for i in queries if i]

#Sentence Transforms
model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
embeddings = model.encode(corpus)

#Output generation

top_k = min(5, len(corpus))
for query in queries:

    query_embedding = model.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print("(Score: {:.4f})".format(score))
        print(corpus[idx], "(Score: {:.4f})".format(score))
        row_dict = Rome.loc[Rome['all_review']== corpus[idx]]
        print("paper_id:  " , row_dict['hotelName'] , "\n")
