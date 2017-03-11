import numpy as np
import pandas as pd
import pyprind
import os

# data from http://ai.stanford.edu/~amaas/data/sentiment/

# loop through IMdB review data subdirectories, read comment text and label and
# save to a DataFrame; progress bar output as file reading takes place
pbar = pyprind.ProgBar(50000)
labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = './aclImdb/%s/%s' % (s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()

# shuffle data in DataFrame and save to CSV file
df.columns = ['review', 'sentiment'] # add columns row
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index=False)
