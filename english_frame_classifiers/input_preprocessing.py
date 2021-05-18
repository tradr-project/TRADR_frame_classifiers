# A script that converts given data to tensors.
# The following data preprocessing steps are made:
# 1) Sentences are tokenized with a BertTokenizerFast.
# 2) Special tokens [CLS] and [SEP] are added to the tokenized sentences.
# 3) For each sentence a position vector is created. The position vector
#    marks target(s) with 1's and the rest of the sentence with 0's.
# 4) Tokenized sentences are converted to the lists of BERT vocabulary IDs.
# 5) For each sentence an attention id vector is created. The attention id vector
#    contains a 1 for each token. It is further padded with 0's.
# 6) A beta vector is created for each sentence. The beta vectors marks the 
#    target as well as certain number of tokens around it with 1's and the
#    rest of the tokens are marked with 0's. 
#    NB: Current WINDOW=10, other appoaches to creating beta vectors are possible.
# 7) Labels (frames) are extracted and converted to numbers.
# 8) Lexical units (LUs) are extracted, and each separate LU is mapped to the
#    set a frames it may evoke based on data. Using these sets filtering mask
#    vectors are created for sentences to filter out frames that are unlikely to be assigned.
# 8) All vectors except labels and filtering masks are padded with 0's.
# 9) All vectors are converted to torch tensors and saved in a given folder.

#import tensorflow as tf
import torch
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizerFast
import pandas as pd
import numpy as np
import ast
from collections import defaultdict
import spacy
import en_core_web_md
from spacy.symbols import ORTH
from spacy.pipeline import Tagger
import sys

flag = sys.argv[1] # tradr or framenet

# Read in the data depending on flag
if flag == 'tradr':
  # Read in TRADR training data, which should consist of 7 columns: speaker, sentence, target indices, lu, frame, parent frame and dialogue act
  # actual column names are ignored
  df = pd.read_csv("./data/tradr_main.tsv", delimiter='\t', header=0, names=['speaker', 'sentence', 'frame', 'parent', 'lu', 'target', 'da'], quoting=3)
elif flag == 'framenet':
  df = pd.read_csv("./data/framenet_main.tsv", delimiter='\t', header=0, names=['sentence', 'target', 'lu', 'frame'], quoting=3)
else:
  print("Type 'tradr' if you want to preprocess TRADR data, and 'framenet' if FrameNet data")


# Remove duplicates
df.drop_duplicates(inplace=True)
print('Data shape:', df.shape)

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# Extract labels (frames) and convert categeries to numbers
lb_make = LabelEncoder()
numerical_labels = lb_make.fit_transform(df.frame.values)
label_tensors = torch.tensor(numerical_labels).long()

# Create a reference dictionary: frame_label --> numerical_value
ref_dict = dict(zip(df.frame.values, numerical_labels))

# Speakers and dialogue acts are specified only for TRADR data
# Convert speakers and dialogue acts to numbers, then store them as tensors
if flag == 'tradr':
  speaker_make = LabelEncoder()
  numerical_speakers = speaker_make.fit_transform(df.speaker.values)
  speaker_tensors = torch.tensor(numerical_speakers).long()

  da_make =LabelEncoder()
  numerical_das = da_make.fit_transform(df.da.values)
  da_tensors = torch.tensor(numerical_das).long()

# Create lu list
lus = df.lu.values

# Create a reference dictionary: LU --> a set of frames this LU evokes
# based ONLY on current data
# NB: It is possible to refine this set using relations between frames!
lu_frames = defaultdict(set)
pairs = zip(lus, df.frame.values)
for lu, frame in pairs:
  lu_frames[lu].add(frame)
  
# For each sentence create mask vectors that can be used to filter the necessary frames
# and convert them to tensors
def create_frame_filtering_mask(frame_set, mask_len):
  if len(frame_set) == 0:
    # filtering is not applied if frame set is empty
    return [1]*mask_len 
  # apply filtering
  mask = [0]*mask_len
  for frame in frame_set:
    idx = ref_dict[frame]
    mask[idx] = 1
  return mask

filtering_masks = [create_frame_filtering_mask(lu_frames[lu], df.frame.nunique()) for lu in lus]
filtering_masks_tensors = torch.tensor(filtering_masks)
print('Built tensors for frame labels, filtering masks, speakers and dialogue acts.')

# Mark targets within the sentences with special markers
target_indices = df.target_indices.values
# In some sentences targets are discontinuous. 
# Store taget indices for all sentences as tuples: [(19, 22), ((0, 2), (3, 8)),...]
target_indices = [ast.literal_eval(target) for target in target_indices]

# Insert special markers for the beginning and end of the targets into each sentence.
# The markers are [TARGET_START] and [TARGET_END]
# Example result: 'They may [TARGET_START] believe [TARGET_END] in apocalyptic prophecy . ' 
# The markers get deleted after having creating position vectors.
sentences = df.sentence_text.values
sentences_marked = []
for i in range(len(sentences)):
  if isinstance(target_indices[i][0], tuple):
    # i.e. the target in the sentence is discontinuous
    parts = []
    counter = 0
    for t in target_indices[i]:
      start = t[0]
      end = t[1]
      parts.extend([sentences[i][counter:start], '[TARGET_START] ', sentences[i][start:end], ' [TARGET_END]'])
      counter = end
    parts.extend(sentences[i][counter:])
    sentences_marked.append(''.join(parts))
  else:
    # the target is a single token
    start = target_indices[i][0]
    end = target_indices[i][1]
    sentences_marked.append(''.join([sentences[i][0:start], '[TARGET_START] ', sentences[i][start:end], ' [TARGET_END]', sentences[i][end:]]))


# SpaCy NLP tools
nlp = en_core_web_md.load()
tagger = Tagger(nlp.vocab)

# Add special case rule
special_case1 = [{ORTH: "[TARGET_START]"}]
special_case2 = [{ORTH: "[TARGET_END]"}]
nlp.tokenizer.add_special_case("[TARGET_START]", special_case1)
nlp.tokenizer.add_special_case("[TARGET_END]", special_case2)

# A list of universal POS tags used by SpaCy (except for 'SPECIAL' and 'PAD' tags).
# Tag 'SPECIAL' is used only for markers '[CLS]' and '[SEP]', and tag 'PAD' - for padded tokens.
pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE', 'SPECIAL', 'PAD']

# Convert POS tags to numbers
pos_encoder = LabelEncoder()
pos_num = pos_encoder.fit_transform(pos_tags)

# Make a reference dict POS tag --> num
pos_to_num = dict(zip(pos_tags, pos_num))

# Pre-tokenize the corpus, tag corpus with POS tags
sentences_tokenized = []
sentences_marked_tokenized = []
sent_pos_tags = [] # list of lists of ints
for idx in range(len(sentences)):
  sent = sentences[idx]
  doc = nlp(sent)
  tokens = [token.text for token in doc]
  pos = [pos_to_num[token.pos_] for token in doc]
  pos = [pos_to_num['SPECIAL']]+pos # add numerical representation of 'SPECIAL' tag
  pos.append(pos_to_num['SPECIAL'])
  sentences_tokenized.append(tokens)
  sent_pos_tags.append(pos)
  sent_marked = sentences_marked[idx]
  doc_marked = nlp(sent_marked)
  tokens_marked = [token.text for token in doc_marked]
  sentences_marked_tokenized.append(tokens_marked)

print('Pre-tokenization with SpaCy is finished.')

# Tokenize pre-tokenized sentences with BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True, additional_special_tokens=['[TARGET_START]', '[TARGET_END]'])
encodings = tokenizer(sentences_tokenized, return_offsets_mapping=True, is_pretokenized=True)
encodings_marked = tokenizer(sentences_marked_tokenized, is_pretokenized=True)

print('BERT tokenization is finished.')

# For each tokenized (original) sentence create a position vector that marks target tokens with 1's and the rest 
# of the tokens with 0's.
tokenized_marked_texts = [tokenizer.convert_ids_to_tokens(i) for i in encodings_marked['input_ids']]
position_vectors = []
for i in range(len(tokenized_marked_texts)):
  position_vector = []
  bit = 0
  for j in range(len(tokenized_marked_texts[i])):
    if tokenized_marked_texts[i][j] == '[TARGET_START]' or tokenized_marked_texts[i][j] == '[TARGET_END]':
      bit = 1-bit # flip bits
    else:
      position_vector.append(bit)
  position_vectors.append(position_vector)

print('Built position vectors.')

# For each position vector extract a list of indices where target(s) occur(s),
# e.g. [0, 0, 0, 1, 1, 0, 0, 0, 0, 0] -> [3,4] and store it in a list
# Target indices are necessary for building alignment vectors later.
target_idx_list = []
for i in range(len(position_vectors)):
  target_idx_list.append([j for j, e in enumerate(position_vectors[i]) if e == 1])

# Extract input ids from BERT encodings
input_ids = encodings['input_ids']

print('Built input ids.')

# assign attention IDs
attention_ids = []
for input_id in input_ids:
    attention_id = [1] * len(input_id)
    attention_ids.append(attention_id)

print('Built attention ids.')

def calc_betas_simple(window, offset, target_list, seq):
  # Calculate betas in the simplest way: for each tensor extract the smallest target position and
  # the largest, beta one is set to the smallest target position minus window (or sequence start+1),
  # beta two is set to the largest position plus window (or sequence end).
  # For sequence seq output a vector, where the elements within betas are ones ans
  # the elements outside these borders are zeroes. This vector shoud be of the same length as seq.
  beta_one = offset # mind the [CLS] token: offset should be 1 if used with BERT
  beta_two = len(seq)-1-offset # mind the [SEP] token: offset should be 1 if used with BERT
  beta_vector = [0]*len(seq)
  for t in target_list:
    beta_one_t = t - window
    beta_two_t = t + window
    if beta_one_t < beta_one:
      beta_one_t = beta_one
    if beta_two_t > beta_two:
      beta_two_t = beta_two
    for i in range(beta_one_t, beta_two_t+1):
      beta_vector[i] = 1
  return beta_vector

# Create beta_vector for each sequence
beta_vectors = [calc_betas_simple(10, 1, target_idx_list[i], input_ids[i]) for i in range(len(input_ids))]

print('Built beta vectors.')

# Set the maximum sentence length
MAX_LEN = max([len(i) for i in encodings['input_ids']])

print('Max sequence length', MAX_LEN, 'is set.')

# Create subtokens masks
is_subword = [(np.array(encodings['offset_mapping'][i])[:,0] != 0).astype(int) for i in range(len(encodings['offset_mapping']))] # list of lists of int (1 == True)
assert(len(is_subword) == len(encodings['input_ids']))

# Extend the POS tags (numerical) to subwords if a word was split by the BERT tokenizer
sent_pos_tags_extended = [] # a list of lists of int
for idx in range(len(is_subword)):
  tags = sent_pos_tags[idx]
  tags_extended = []
  idx_counter = -1
  for subword in is_subword[idx]:
    if subword == 0:
      idx_counter += 1
    curr_tag = tags[idx_counter]
    tags_extended.append(curr_tag)
  sent_pos_tags_extended.append(tags_extended)

print('Built feature vectors.')

# Pad our input tokens, position vectors, attention ids, beta vectors and additional feature vectors
input_ids_padded = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
position_vectors_padded = pad_sequences(position_vectors, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
attention_ids_padded = pad_sequences(attention_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
beta_vectors_padded = pad_sequences(beta_vectors, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
sent_pos_tags_extended_padded = pad_sequences(sent_pos_tags_extended, maxlen=MAX_LEN, dtype="long", value=pos_to_num['PAD'], truncating="post", padding="post")
is_subword_padded = pad_sequences(is_subword, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Convert inputs to PyTorch tensors
text_tensors = torch.tensor(input_ids_padded)
attention_tensors = torch.tensor(attention_ids_padded)
position_tensors = torch.tensor(position_vectors_padded)
beta_tensors = torch.tensor(beta_vectors_padded)
pos_tags_tensors = torch.tensor(sent_pos_tags_extended_padded)
subword_tensors = torch.tensor(is_subword_padded)

# Save everything
data_output = "./train_data/"+flag+"/"+flag+"_" # e.g., "./train_data/tradr/tradr_"
num_cls = "class_"+str(df.frame.nunique()) # e.g., "class_81"

torch.save(label_tensors, data_output+"label_tensors_"+num_cls+"_len_"+str(MAX_LEN)+".pt")
torch.save(text_tensors, data_output+"text_tensors_"+num_cls+"_len_"+str(MAX_LEN)+".pt")
torch.save(attention_tensors, data_output+"attention_tensors_"+num_cls+"_len_"+str(MAX_LEN)+".pt")
torch.save(position_tensors, data_output+"position_tensors_"+num_cls+"_len_"+str(MAX_LEN)+".pt")
torch.save(beta_tensors, data_output+"beta_tensors_"+num_cls+"_len_"+str(MAX_LEN)+".pt")
torch.save(filtering_masks_tensors, data_output+"filtering_masks_tensors_"+num_cls+"_len_"+str(MAX_LEN)+".pt")
torch.save(pos_tags_tensors, data_output+"pos_tags_tensors_"+num_cls+"_len_"+str(MAX_LEN)+".pt")
torch.save(subword_tensors, data_output+"subword_tensors_"+num_cls+"_len_"+str(MAX_LEN)+".pt")
# Speaker and dialogue act tags are provided only for TRADR data:
if flag == 'tradr':
  torch.save(speaker_tensors, data_output+"speaker_tensors_"+num_cls+"_len_"+str(MAX_LEN)+".pt")
  torch.save(da_tensors, data_output+"da_tensors_"+num_cls+"_len_"+str(MAX_LEN)+".pt")
