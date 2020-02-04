# trending_topics - find trending words from news

Toward a sophisticated and efficient way of finding words which are trending in news

## Usage
### Readers
To read the analysis please refer to the Notebook [SUBMISSION Finding ...](https://github.com/md-nlp-experiments/trending_topics/blob/master/SUBMISSION%20Finding%20Trending%20Words%20in%20News.ipynb)

### Developers
To run the code:
- clone the repo
- install `requirements.txt`
- place the raw data file `rna002_RTRS_2017_11_29.csv` in directory raw_data/

Folder structure should be as follows:

	    .
	    ├── ground_truth/
	    ├── source/
	    ├── raw_data/rna002_RTRS_2017_11_29.csv
	    ├── ...
    
- Lines 1-3 in file `source/load_libs.py` is used to specify environment locations should this be manually needed. 
Edit or remove appropriately (likely simply removing this should work)
- You can now run the SUBMISSION notebook where all activities are further described
