# A script that reads in German TRADR and SALSA data, removes duplicates and rare frames (those that have less
# than 5 instances), splits TRADR data into parts designed for training (tradr_main.tsv) and testing
# (tradr_test_a.tsv and tradr_test_b.tsv) and takes a subset of SALSA data (salsa_for_sampling.tsv)
# that contains only instances of frames that occur in TRADR.

import pandas as pd
from collections import defaultdict

# Read in TRADR data (discard 'Parent frame(s)' column)
df_tradr = pd.read_csv("./data/tradr_ger_from_conll_format.txt", delimiter='\t', header=None, names=['sentence', 'frame', 'lu', 'sentence_marked'], quoting=3, usecols=[0, 1, 3, 4])
print('TRADR data size before dropping dupes:', df_tradr.shape)

# Read in SALSA data (discard 'Index' column)
df_salsa = pd.read_csv("./data/salsa_data_with_targets_consistent.tsv", delimiter='\t', header=None, names=['sentence', 'frame', 'lu', 'sentence_marked'], quoting=3, usecols=[1, 2, 3, 4])
print('Salsa data size before dropping dupes:', df_salsa.shape)

# Remove duplicates from data
df_tradr.drop_duplicates(inplace=True)
df_salsa.drop_duplicates(inplace=True)
print('TRADR data size after removing dupes:', df_tradr.shape)
print('Salsa data size after removing dupes:', df_salsa.shape)

# Shuffle the data
df_tradr = df_tradr.sample(frac=1).reset_index(drop=True)
df_salsa = df_salsa.sample(frac=1).reset_index(drop=True)

# Split into part reserved only for testing and main part
sep_tradr = int(df_tradr.shape[0]*0.1) # take 10% for testing
df_tradr_test = df_tradr.iloc[:sep_tradr]
df_tradr_main = df_tradr.iloc[sep_tradr:]

print('TRADR main data size:', df_tradr_main.shape)
print('TRADR initial test data size:', df_tradr_test.shape)
print('Number of frame labels in TRADR data before removing rare frames:', df_tradr_main.frame.nunique())

# Remove from both datasets frames that occur less than 5 times
# This is needed to perform cross validation later
df_tradr_main = df_tradr_main[df_tradr_main['frame'].map(df_tradr_main['frame'].value_counts()) > 4]
df_tradr_main = df_tradr_main.reset_index(drop=True)
print('TRADR main data size after removing rare frames:', df_tradr_main.shape)
print('Number of frame labels in TRADR data:', df_tradr_main.frame.nunique())

# Create a subset of SALSA data which has only those frame labels that are
# present in TRADR main data (after rare frames have been removed)
df_salsa_for_sampling = df_salsa[df_salsa['frame'].isin(df_tradr_main.frame.values)]
print('Salsa subset for sampling size:', df_salsa_for_sampling.shape) 

# If testing sets contain any frame labels that are not in the training part,
# instances with these labels get removed
df_tradr_test_a = df_tradr_test[df_tradr_test['frame'].isin(df_tradr_main.frame.values)] # main TRADR testing set
print('TRADR testing data A size:', df_tradr_test_a.shape)

# Create subset of 'df_tradr_test_a' that does not contain the instances of frames 'Communication_by_protocol',
# 'Communication_response_message' and 'Communication_fragment'
df_tradr_temp = df_tradr_test_a.loc[(df_tradr_test_a['frame']!='Communication_by_protocol')]
df_tradr_temp = df_tradr_temp.loc[(df_tradr_temp['frame']!='Communication_response_message')]
df_tradr_test_b = df_tradr_temp.loc[(df_tradr_temp['frame']!='Communication_fragment')]
print('TRADR testing data B size:', df_tradr_test_b.shape)

# Store both test and main part as .TSV files
df_tradr_test_a.to_csv("./data/tradr_test_a.tsv", sep='\t', index=False)
df_tradr_test_b.to_csv("./data/tradr_test_b.tsv", sep='\t', index=False)
df_tradr_main.to_csv("./data/tradr_main.tsv", sep='\t', index=False)
df_salsa_for_sampling.to_csv("./data/salsa_for_sampling.tsv", sep='\t', index=False)
