# A Python script to extract data from FrameNet corpus.
# We are interested in the following entries:
# a sentence | target indices (possibly several of them, comma separated, e.g. [(44, 47),(59, 68)] | lexical unit | frame (label)
# all entries (tab separated) are stored as a line in a .tsv file

import nltk
from nltk.corpus import framenet as fn

def extract_from_framenet():
  with open('./data/framenet_data_with_LU.tsv', 'w') as f:
    sents = fn.exemplars()
    for sent in sents:
      try:
        text = sent['text']
        target_indices = str(sent['Target']).strip('[]')
        lu = sent['LU']['name']
        frame_name = sent['frame']['name']
        f.write('\t'.join([text, target_indices, lu, frame_name])+'\n')
      except KeyError:
        # a sentence may not have key 'Target'
        continue

extract_from_framenet()

