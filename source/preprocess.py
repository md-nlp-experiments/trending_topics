
from load_libs import *

def make_plt(df,key_list, top_list):
    #key_list=[list(par.keys())[0] for par in parameters]
    nr_plt_per_row=5
    fig, axs = plt.subplots(int(np.ceil(len(key_list)/nr_plt_per_row)),nr_plt_per_row, 
                            figsize=(20, int(4*np.ceil(len(key_list)/nr_plt_per_row))), 
                            facecolor='w', edgecolor='k')

    axs = axs.ravel()
    for i , key in enumerate(key_list):
        clr='r' if key in top_list else 'b' 
        axs[i].plot(df.index.values,df[key].values,label='Frequency', color=clr)
        axs[i].set_title(key) 

        #if i == len(key_list)-1:
        #    axs[i].legend()
    plt.setp(axs, yticks=np.linspace(0,4,5))

def summary_stats(df, max_smpl=10):
    '''
    Provides basic summary of a dataframe beyond .describe.
    Helps mostly with raw, text and/or categorical data.
    Answers: % missing values, number of unique values & sample values for each feature
    '''
    df_desc=pd.DataFrame([],index=df.columns)
    df_desc['Unique']=df.nunique()
    df_desc['# Missing']=[sum(df[col_nm].isin(['',np.nan])) for col_nm in df.columns]
    df_desc['# Zeros']=[sum(df[col_nm].isin([0,'0'])) for col_nm in df.columns]

    df_desc['Available']=(len(df)-df_desc['# Missing']-df_desc['# Zeros'])
    df_desc['% Available']=100*((df_desc['Available']/len(df)).round(2))
    df_desc['Types']=df.dtypes

    df_desc['Sample']=[sample(set(df[col_nm].fillna('')),min(max_smpl,len(set(df[col_nm])))) for col_nm in df.columns]

    return df_desc


def rm_rows_starting(df,column_nm,list_to_drop):
    ''' Remove rows where the string content starts with any of the entries in list_to_drop '''
    def bool_keep_instance(instance,list_to_drop):
        ''' True if we keep the instance, ie not in list. Else False '''
        return not max([instance[:len(i)]==i for i in list_to_drop])
    
    title_to_keep=df[column_nm].apply(lambda x: bool_keep_instance(x,list_to_drop))
    res=df.copy()
    res=res[title_to_keep]
    return res

def rm_values_by_flag(df, column_nm, list_to_drop):
    ''' Generic to remove rows where certain conditions apply on a given column '''
    if not isinstance(list_to_drop,list):
        list_to_drop=[list_to_drop]
    res=df.copy()
    rows_to_drop=res[column_nm].isin(list_to_drop)
    return res[~rows_to_drop]

def mrg_cols(df,cols_to_mrg,new_col):
    ''' Merges two text colums into one new one'''
    assert(isinstance(cols_to_mrg,list))
    assert(len(cols_to_mrg)==2)
    res=df.copy()
    res[new_col]=res.apply(lambda x: x[cols_to_mrg[0]] + ' ' + x[cols_to_mrg[1]], axis=1)
    return res

def clean_txt(df,txt_to_clean):
    ''' Basic text cleaner. Removes tabs and new lines from text column txt_to_clean'''
    res=df.copy()
    res[txt_to_clean]=res[txt_to_clean].apply(lambda x: x.replace('\t','').replace('\n','').replace('--',''))
    return res

def group_by_id(df, select_cols, sort_by, group_by):
    ''' 
    Selects the first value of a group by and sorts by sort_by column first where needed
    In this application, not all article bodies are populated so sorting by those helps pick unique articles with body
    '''
    res=df.copy()
    res=res[select_cols]
    res=res.sort_values(sort_by , ascending=False)
    res=res.groupby(group_by, as_index=False,).first()    
    return res

