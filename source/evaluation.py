from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import random 
from source.modelling import TimeBuckets

def t_test_dist(df_emb,df_idx,instance,cos_dist_full):
    # Taking article index from the word dataframe and the tfidf embedding from the "article body" one
    df_instance=df_emb[df_idx[instance]!=0]
    cos_res_instance=cosine_similarity(df_instance,  df_instance)
    cos_dist_instance=cos_res_instance[np.triu_indices_from(cos_res_instance,k=1)]
    cos_dist_inst_rept=np.random.choice(cos_dist_instance, len(cos_dist_full), replace=True)
    _, p2 = stats.ttest_ind(cos_dist_full,cos_dist_inst_rept)
    return p2

def pipe_transform(df,pipe_ttl,time_interval):
    # df_flat_indicators - dataframe from the same experiment but not aggregated into time buckets. 
    #    We will use this one to calculate cosine similarities between articles belonging to the same trending word
    df_flat_indicators=pipe_ttl.transform(list(df[pipe_ttl.col_nm]))

    # NOTE: currently set as 'sum' for word counts and 'max' for tfidf
    time_bkts=TimeBuckets(df_index=list(df['STORY_DATE_TIME']), 
                             aggr_approach=pipe_ttl.aggr_approach, 
                             time_interval=time_interval)
    #df_aggregate - above dataframe aggregated by time buckets
    df_aggregate=time_bkts.transform(df_flat_indicators)
    return df_flat_indicators, df_aggregate

def get_ground_truths(path):
    # Load data from file and generate lists of all topics and ground truths
    df_lbl=pd.read_csv(path, index_col=0)
    all_titles=list(df_lbl.Topic)
    trending_titles=list(df_lbl[df_lbl.Label==1].Topic)
    return all_titles, trending_titles, list(df_lbl.Label)

def pipe_predict(predict_list,df, pipe_ttl,pipe_art, time_interval=5):
    '''
    The prediction function for the pipeline for the current experiment

    Inputs:
    predict_list - list of words or ngrams we are predicting
    df - as returned by the preprocessing pipeline
    time_interval - relevant time interval for aggregation of results
    pipe_ttl - fitted pipeline convering the TITLES from df into a time series of word counts/frequencies
    pipe_art - fitted pipeline convering the ARTICLES from df into a time series of word counts/frequencies
     
    '''

    # df_flat_indicators - dataframe from the same experiment but not aggregated into time buckets. 
    #    We will use this one to calculate cosine similarities between articles belonging to the same trending word
    #df_aggregate - above dataframe aggregated by time buckets
    df_flat_indicators, df_aggregate= pipe_transform(df,pipe_ttl,time_interval)
        
    # df_flat_eval - these are the actual article vectors we want to calculate cosine similarity on, 
    #    ie these will be likely be tfidf/counts from the article body while the previous tables where from the article titles
    #    While we use titles to find trending words, we use article bodies to calculate cosine similarity as the article body will be richer for a cosine similarity
    #    Therefore the column names may not match with the previous tables as the corpus here is completely different   
    df_flat_eval=pipe_art.transform(list(df[pipe_art.col_nm]))
    #print(df_aggregate.columns)
    # Ensure topics can be found in the predicted data
    assert(len(set(predict_list)&set(df_aggregate.columns))==len(predict_list))
    # Ensure all data tables have the same column names, ie come from the same experiments
    assert(set(df_aggregate.columns)==set(df_flat_indicators.columns))

    pred_lbl=[]
    topic_ttls=[]
    pred_ttest=[]
    pred_adf=[]

    cos_res_full=cosine_similarity(df_flat_eval,  df_flat_eval)
    cos_dist_full=cos_res_full[np.triu_indices_from(cos_res_full,k=1)]

    for wrd in predict_list:
        adf_res=adfuller(df_aggregate[wrd].rolling(24).sum().fillna(0), autolag='AIC')[1]
        t_res=t_test_dist(df_flat_eval,df_flat_indicators,wrd,cos_dist_full)
        pred_ttest.append(t_res)
        pred_adf.append(adf_res)
        if (t_res<0.05) & (adf_res>.05):
            pred_lbl.append(1)
            topic_ttls.append(wrd)
        else:
            pred_lbl.append(0)

    df_res=pd.DataFrame([],index=predict_list,)
    df_res['Prediction']=pred_lbl
    df_res['T_stat']=pred_ttest
    df_res['ADF_stat']=pred_adf
    return  pred_lbl, df_res