import sys
sys.path
sys.path.append('/Users/md/Downloads/trending_topics/env-trending/lib/python3.7/site-packages')

import pandas as pd
from random import choice, sample
import numpy as np
import csv 
import datetime
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', 500)

import os
import spacy
import string

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

from gensim import corpora, models
import gensim

'''
# Had some issues finding spacy in my after installing to environment, 
# so this adds the environment relevant for this exercise to sys.path

import sys
sys.path
sys.path.append('/Users/md/Downloads/trending_topics/env-trending/lib/python3.7/site-packages')

import pandas as pd
from random import choice, sample
import numpy as np
import csv 
import datetime
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', 500)

import os
import spacy
import string

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

from gensim import models
import gensim

import matplotlib.pyplot as plt'''