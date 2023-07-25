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
import tqdm
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
        papers_to_process = pd.concat([papers_to_process, pd.read_csv(os.getcwd().replace('scripts', 'data') + f'/text_corpus/text_corpus{i}.csv').loc[:100]])

    papers_to_process = papers_to_process.drop_duplicates().reset_index(drop = True)
    
    print(papers_to_process.isnull().values.any())

    processed_docs = papers_to_process['text'].map(preprocess)
    id2word = corpora.Dictionary(processed_docs)
    bow_corpus = [id2word.doc2bow(doc) for doc in processed_docs]

    grid = {}
    grid["Validation_Set"] = {}

    # Topics range
    min_topics = 2
    max_topics = 11
    step_size = 1
    topics_range = range(min_topics, max_topics, step_size)

    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append("symmetric")
    alpha.append("asymmetric")

    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append("symmetric")


    def compute_coherence_values(corpus, texts, k, a, b):
        print("model")
        lda_model = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=corpora.Dictionary(texts),
            num_topics=k,
            random_state=100,
            chunksize=100,
            passes=10,
            alpha=a,
            eta=b,
        )
        print("coherence")
        coherence_model_lda = CoherenceModel(
            model=lda_model, corpus=corpus, texts = texts, coherence="c_v"
        )

        return coherence_model_lda.get_coherence()

    def runTuning(corpus, docs):
        num_of_docs = len(corpus)
        corpus_sets = [
                gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)),
                corpus,
            ]

        corpus_title = ["75% Corpus", "100% Corpus"]

        model_results = {
                "Validation_Set": [],
                "Topics": [],
                "Alpha": [],
                "Beta": [],
                "Coherence": [],
            }
            # Can take a long time to run
        if 1 == 1:
            print("pbar")
            pbar = tqdm.tqdm(
                total=(len(beta) * len(alpha) * len(topics_range) * len(corpus_title))
                )
            print("pbar done")

                # iterate through validation corpuses
            for i in range(len(corpus_sets)):
                    # iterate through number of topics
                for k in topics_range:
                        # iterate through alpha values
                    for a in alpha:
                            # iterare through beta values
                        for b in beta:
                                # get the coherence score for the given parameters
                            cv = compute_coherence_values(
                                    corpus=corpus[i], texts = docs, k=k, a=a, b=b
                                )
                                # Save the model results
                            model_results["Validation_Set"].append(corpus_title[i])
                            model_results["Topics"].append(k)
                            model_results["Alpha"].append(a)
                            model_results["Beta"].append(b)
                            model_results["Coherence"].append(cv)

                            pbar.update(1)
            pd.DataFrame(model_results).to_csv(
                    "./results/lda_tfidf_tuning_results.csv", index=False
                )
            pbar.close()

    runTuning(bow_corpus,processed_docs)

    # number of topics
    # num_topics = 4
    # # Build LDA model
    # lda_model = gensim.models.LdaMulticore(corpus=bow_corpus,
    #                                     id2word=id2word,
    #                                     num_topics=num_topics)

    # LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))

    # if 1 == 1:
    #     LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, bow_corpus, id2word)
    #     with open(LDAvis_data_filepath, 'wb') as f:
    #         pickle.dump(LDAvis_prepared, f)
    # # load the pre-prepared pyLDAvis data from disk
    # with open(LDAvis_data_filepath, 'rb') as f:
    #     LDAvis_prepared = pickle.load(f)
    # pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html')

    
