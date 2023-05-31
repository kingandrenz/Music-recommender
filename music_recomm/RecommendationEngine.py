
%matplotlib inline
import pandas as pd
import numpy as np
from numpy import int64
​
import requests
import IPython.display as Disp
import sklearn
from sklearn.decomposition import TruncatedSVD



Read dataset that shows metadata of each song into pandas dataframe
dataset_url = r"C:\Users\HP\Downloads\Music Info.csv\Music Info.csv"

song_df= pd.read_csv(dataset_url)
song_df.head()
