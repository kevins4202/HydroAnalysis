import gensim
import gensim.corpora as corpora
from gensim import models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from nltk.stem.porter import *
stemmer = SnowballStemmer('english')
import numpy as np
np.random.seed(2018)
import nltk
import logging
import os
import requests as re
import pandas as pd
# nltk.download('wordnet')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from pprint import pprint
import pyLDAvis.gensim
import pickle 
import pyLDAvis
from wordcloud import WordCloud
# nltk.download('omw-1.4')

if __name__=="__main__":
    stop_words = stopwords.words('english')

    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) 
                if word not in stop_words] for doc in texts]

    def lemmatize_stemming(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        return result

    papers_to_process = pd.DataFrame(columns = ['index', 'text'])

    for i in range(1):
        papers_to_process = pd.concat([papers_to_process, pd.read_csv(f'/Volumes/mac/corpus/text_corpus{i}.csv').loc[:]])

    papers_to_process = papers_to_process.drop_duplicates().reset_index(drop = True)

    processed_docs = papers_to_process['text'].map(preprocess)
    id2word = corpora.Dictionary(processed_docs)
    bow_corpus = [id2word.doc2bow(doc) for doc in processed_docs]

    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    # number of topics
    num_topics = 4
    # Build LDA model
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=3, id2word=id2word, passes=2, workers=4)

    LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))

    #tfidf
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model_tfidf, bow_corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_tfidf_prepared_'+ str(num_topics) +'.html')
    
    coherence_model_lda = CoherenceModel(model=lda_model_tfidf, texts=processed_docs, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score: ', coherence_lda)