import pandas as pd
import string

# Read in TRADR data
df = pd.read_csv("./data/salsa_data_with_targets_inconsistent.txt", delimiter='\t', header=None, names=['sentence', 'frame', 'lu', 'sentence_marked'], quoting=3)
print("Original shape:", df.shape)

df.drop_duplicates(inplace=True)
print("After deleting dupes:", df.shape)

# Remove inconsistencies: 
df = df.drop_duplicates(['sentence_marked'], keep='first').reset_index(drop=True)

print("After deleting inconsistencies:", df.shape)

df_upd =  df['frame']!='Unannotated'
df = df[df_upd].reset_index(drop=True)
print("After deleting instances of 'Unannotated' frame:", df.shape)

df.to_csv("./data/salsa_data_with_targets_consistent.tsv", sep='\t', index=False, header=False)
