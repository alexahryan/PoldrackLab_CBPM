import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
%matplotlib inline
import pandas as pd
import seaborn as sns
from pathlib import Path
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#the input to this code will be a set of fc matrices, stored as .txt or .csv files in a directory called fc_data
top_dir = Path("./")
data_dir = top_dir/"fc_data/"
#reading in the sample
#need to change the title of the text file
subj_list = pd.read_csv('DATA.txt', header=None)
subj_list = np.array(subj_list, dtype=str).flatten()
#reading in behavioral data
#need to change the title of file
all_behav_data = pd.read_csv('BEHAVIORAL_DATA.csv', dtype={'Subject': str})
all_behav_data.set_index('Subject', inplace=True)
print(all_behav_data.shape)
all_behav_data.head()