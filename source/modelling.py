
from source.load_libs import *

def round_time(date_time,interval):
    ''' Helper to round a datetime string to the closest rounding interval, eg to the closest 5min if interval is 5min '''
    dt=datetime.datetime.strptime(date_time,'%Y-%m-%d %H:%M:%S')
    out=str(datetime.datetime(dt.year,dt.month,dt.day,dt.hour,interval*int(dt.minute/interval),0 ))
    out=out.replace('-','').replace(':','').replace(' ','_')
    return out


punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS
parser = English()
# Custom transformer using spaCy

class predictors(TransformerMixin):
    # Transforms all to lower case
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [text.strip().lower() for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

class TimeBuckets(TransformerMixin):
    # Aggregates data into time buckets
    def __init__(self, df_index, aggr_approach, time_interval):
        
        self.df_index = df_index
        self.time_interval = time_interval
        self.aggr_approach = aggr_approach.lower()

    def transform(self, X, **transform_params):

        df_cnt_wrd=X.copy()
        # Createa Time bucket column
        df_cnt_wrd['TAKE_DATE_TIME']=list(self.df_index)
        df_cnt_wrd['TIME_BUCKET']=df_cnt_wrd['TAKE_DATE_TIME'].apply(lambda x:round_time(x,self.time_interval))
        df_cnt_wrd=df_cnt_wrd.drop('TAKE_DATE_TIME',axis=1)
        
        # Group by dataframe based on aggregation approach
        if self.aggr_approach=='sum':
            df_bkts=df_cnt_wrd.groupby('TIME_BUCKET').sum()
        elif self.aggr_approach=='max':
            df_bkts=df_cnt_wrd.groupby('TIME_BUCKET').max()
        df_bkts=df_bkts.sort_index()
        return df_bkts

    def fit(self, X, y=None, **fit_params):
        return self


class LDAVectorizer(TransformerMixin):
    # Wraps the GenSim LDA model into a sklearn transformer format in order to use interchangeably into modelling pipeline
    def __init__(self, topic_wrd_cutoff=0.1,lda_topics=30, lda_passes=10,lda_workers=4):
        self.nr_topics=lda_topics
        self.passes=lda_passes
        self.workers=lda_workers
        self.topic_wrd_cutoff=topic_wrd_cutoff
    
    def build_grid_for_LDA(self,a,nr_topics):
        zz=np.zeros([len(a),nr_topics])

        for idx in range(len(a)):
            for e in a[idx]:
                zz[idx,e[0]]=e[1]
        return zz

    def transform(self, X, **transform_params):
        pre_process=[spacy_tokenizer(sent) for sent in X]
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in pre_process]
        corpus_tfidf = self.tfidf[bow_corpus]
        res_gensim=[self.lda_model_tfidf[sent] for sent in corpus_tfidf] 
        
        # Make table of all LDA results
        lda_table=self.build_grid_for_LDA(res_gensim,self.nr_topics)

        # Make table of topic names from topic descriptions
        topic_table=[[wrd.split('*') for wrd in self.lda_model_tfidf.print_topic(idx).split(' + ')] for idx in range(self.nr_topics)]
        topic_titles=["_".join([j[1:-1] for i, j in topic_table[idx] if float(i)>self.topic_wrd_cutoff]) for idx in range(self.nr_topics)]
        topic_titles=[title if title!='' else 'topic_'+str(i) for (i, title) in enumerate(topic_titles)]
        # Output DataFrame format consistent with other 
        lda_table_df=pd.DataFrame(lda_table, columns=topic_titles)
        return lda_table_df

    def fit(self, X, y=None, **fit_params):
        # Tokenize via spacy. Tokenizer is not internalized to the class as serializing the spacy tokenizer takes too much space
        pre_process=[spacy_tokenizer(sent) for sent in X]
        # Work out reuable model dictionary
        self.dictionary = gensim.corpora.Dictionary(pre_process)
        self.dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        # Training set specific corpus
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in pre_process]

        self.tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = self.tfidf[bow_corpus]
        self.lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, 
                                             num_topics=self.nr_topics, id2word=self.dictionary, passes=self.passes, workers=self.workers)
        return self

class Vectorizer(TransformerMixin):
    # Wrapper class around different types of vectorizers
    # A generic vectorizer abstracts away this part of the pipeline to speed up the experimentation loop
    # Unlike the raw Vectorizers, this class outputs a dataframe where each column is the topic/word from the raw Vactorizer. 
    # This last step is done for convenience in the follow on analytics steps
    def __init__(self, vect_approach, tokenizer,ngram_range, min_df, max_df, lda_topics):
        self.vect_approach=vect_approach.lower()
        if self.vect_approach=='count':
            self.vectr = CountVectorizer(tokenizer = tokenizer, ngram_range=ngram_range,min_df=min_df,max_df=max_df)
        elif self.vect_approach=='tfidf':
            self.vectr = TfidfVectorizer(tokenizer = tokenizer,ngram_range=ngram_range,min_df=min_df,max_df=max_df)
        elif self.vect_approach=='lda':
            # Note, LDA values hardcoded for now except nr topics
            self.vectr = LDAVectorizer(topic_wrd_cutoff=0.1,lda_topics=lda_topics, lda_passes=10,lda_workers=4)
        else:
            print('Vectorizer approach either "lda", "tfidf" or "count"')

    def transform(self, X, **transform_params):
        
        if self.vect_approach in ['count','tfidf']:
            X=self.vectr.transform(X)
            vocab=self.vectr.vocabulary_
            word_index=pd.DataFrame(list(vocab.keys()),index=list(vocab.values()), columns=['term'])
            word_index=word_index.sort_index()['term']

            df_cnt_wrd=pd.DataFrame(X.A,columns=word_index)
        elif self.vect_approach=='lda':
            # Note, LDA Vectorizer defined above already transforms results into a dataframe with topics as columns 
            df_cnt_wrd=self.vectr.transform(X)
        return df_cnt_wrd

    def fit(self, X, y=None, **fit_params):
        self.vectr=self.vectr.fit(X)
        return self

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens




class TrainPipe():
    '''
    Purpose here is to generate word counts or tf-idf for a select column from a dataframe.
    - df - Pandas DataFrame with texts and a 'TAKE_DATE_TIME' with datetimes
    - col_nm: name of column to be used
    - vect_approach: takes "count" of "tfidf" 
    Returns:
    - Dataframe containing time bucketed timeseries of ngrams built acccording to perscribed pipeline
    - DataFrame of the same transformations before the aggregation up to time buckets
    - the trained pipeline before time bucketing aggregation
    '''
    
    def __init__(self,col_nm,vect_approach,tokenizer=spacy_tokenizer,ngram=2,min_df=5,max_df=0.6, lda_topics=30):
        predictor=predictors()
        vectr=Vectorizer(vect_approach, tokenizer,ngram_range=(1,ngram), min_df=min_df, max_df=max_df, lda_topics=lda_topics)

        self.pipe = Pipeline([("cleaner", predictor),
                        ('vectorizer', vectr),
                        ])
        self.vect_approach=vect_approach.lower()
        if self.vect_approach=='count':
            self.aggr_approach='sum'
        elif self.vect_approach in ['tfidf', 'lda']:
            self.aggr_approach='max'
        else:
            print('Vectorizer approach either "lda", "tfidf" or "count"')   
 
        self.col_nm=col_nm
    
    def fit(self, df, y=None, **fit_params):
        rnd_input=sample(list(df[self.col_nm]),len(df))
        
        self.pipe.fit(rnd_input)

        return self

    def transform(self, X, y=None, **fit_params):

        return self.pipe.transform(X)