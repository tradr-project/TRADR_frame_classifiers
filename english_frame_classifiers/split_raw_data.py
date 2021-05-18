import pandas as pd
from collections import defaultdict

# Read in TRADR data
df_tradr = pd.read_csv("./data/tradr_data_for_clf_clean.tsv", delimiter='\t', header=None, names=['speaker', 'sentence', 'frame', 'parent', 'lu', 'target', 'da'], quoting=3)
print('TRADR data size before dropping dupes:', df_tradr.shape)

df_framenet = pd.read_csv("./data/framenet_data_with_LU.tsv", delimiter='\t', header=None, names=['sentence', 'target', 'lu', 'frame'], quoting=3, usecols=[0, 1, 2, 3])
print('FrameNet data size before dropping dupes:', df_framenet.shape)

# Remove duplicates from data
df_tradr.drop_duplicates(inplace=True)
df_framenet.drop_duplicates(inplace=True)
print('TRADR data size after removing dupes:', df_tradr.shape)
print('FrameNet data size after removing dupes:', df_framenet.shape)

# Shuffle the data
df_tradr = df_tradr.sample(frac=1).reset_index(drop=True)
df_framenet = df_framenet.sample(frac=1).reset_index(drop=True)

# Split into part reserved only for testing and main part
sep_tradr = int(df_tradr.shape[0]*0.1) # take 10% for testing
sep_framenet = int(df_framenet.shape[0]*0.1)
df_tradr_test = df_tradr.iloc[:sep_tradr]
df_tradr_main = df_tradr.iloc[sep_tradr:]

df_framenet_test = df_framenet.iloc[:sep_framenet]
df_framenet_main = df_framenet.iloc[sep_framenet:]
print('TRADR main data size:', df_tradr_main.shape)
print('TRADR initial test data size:', df_tradr_test.shape)
print('FrameNet main data size:', df_framenet_main.shape)
print('FrameNet initial test data size:', df_framenet_test.shape)
print('Number of frame labels in TRADR data before removing rare frames:', df_tradr_main.frame.nunique())
print('Number of frame labels in FrameNet data before removing rare frames:', df_framenet_main.frame.nunique())

# Remove from both datasets frames that occur less than 5 times
# This is needed to perform cross validation later
df_tradr_main = df_tradr_main[df_tradr_main['frame'].map(df_tradr_main['frame'].value_counts()) > 4]
df_tradr_main = df_tradr_main.reset_index(drop=True)
df_framenet_main = df_framenet_main[df_framenet_main['frame'].map(df_framenet_main['frame'].value_counts()) > 4]
df_framenet_main = df_framenet_main.reset_index(drop=True)
print('TRADR main data size after removing rare frames:', df_tradr_main.shape)
print('FrameNet main data size after removing rare frames:', df_framenet_main.shape)
print('Number of frame labels in TRADR data:', df_tradr_main.frame.nunique())
print('Number of frame labels in FrameNet data:', df_framenet_main.frame.nunique())

# Create a subset of FrameNet data which has only those frame labels that are
# present in TRADR main data (after rare frames have been removed)
df_framenet_for_sampling = df_framenet[df_framenet['frame'].isin(df_tradr_main.frame.values)]
print('FrameNet subset for sampling size:', df_framenet_for_sampling.shape) 

# If testing sets contain any frame labels that are not in the training part,
# instances with these labels get removed
df_tradr_test_a = df_tradr_test[df_tradr_test['frame'].isin(df_tradr_main.frame.values)] # main TRADR testing set
print('TRADR testing data A size:', df_tradr_test_a.shape)

df_framenet_test = df_framenet_test[df_framenet_test['frame'].isin(df_framenet_main.frame.values)]
print('FrameNet testing data size:', df_framenet_test.shape)

# Create subsets of 'df_tradr_test_a' 
# 1) one that does not contain the instances of frames 'Communication_by_protocol',
# 'Communication_response_message' and 'Communication_fragment'
df_tradr_temp = df_tradr_test_a.loc[(df_tradr_test_a['frame']!='Communication_by_protocol')]
df_tradr_temp = df_tradr_temp.loc[(df_tradr_temp['frame']!='Communication_response_message')]
df_tradr_test_b = df_tradr_temp.loc[(df_tradr_temp['frame']!='Communication_fragment')]
print('TRADR testing data B size:', df_tradr_test_b.shape)
# 2) one that is suitable for testing models trained purely on FrameNet, 
# i.e. one that does not contain instances of frames absent in FrameNet training data
df_tradr_test_c = df_tradr_test_a[df_tradr_test_a['frame'].isin(df_framenet_main.frame.values)]
print('TRADR testing data C size:', df_tradr_test_c.shape)

# Store both test and main part as .CSV files
df_tradr_test_a.to_csv("./data/tradr_test_a.tsv", sep='\t', index=False)
df_tradr_test_b.to_csv("./data/tradr_test_b.tsv", sep='\t', index=False)
df_tradr_test_c.to_csv("./data/tradr_test_c.tsv", sep='\t', index=False)
df_tradr_main.to_csv("./data/tradr_main.tsv", sep='\t', index=False)
df_framenet_test.to_csv("./data/framenet_test.tsv", sep='\t', index=False)
df_framenet_main.to_csv("./data/framenet_main.tsv", sep='\t', index=False)
df_framenet_for_sampling.to_csv("./data/framenet_for_sampling.tsv", sep='\t', index=False)
