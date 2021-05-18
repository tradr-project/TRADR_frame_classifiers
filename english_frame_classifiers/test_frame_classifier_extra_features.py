# A script to evaluate the performance of the model
# on test data

import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, AutoModel, BertTokenizerFast
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from imblearn.metrics import classification_report_imbalanced
from progressbar import ProgressBar
import sys

test_name = sys.argv[1] # should be of form 'tradr_test_a' or 'tradr_test_b', etc.

# Read in all the data
df = pd.read_csv("./data/tradr_main.tsv", delimiter='\t', header=0, names=['speaker', 'sentence', 'frame', 'parent', 'lu', 'target', 'da'], quoting=3)

# Extract labels (frames) and convert categeries to numbers
# Labels are only needed for classification report
lb_make = LabelEncoder()
numerical_labels = lb_make.fit_transform(df.frame.values)

# Create a reference dictionary: numerical_value --> frame label
ref_dict = dict(zip(numerical_labels, df.frame.values))

# Number of labels
num_frames = df.frame.nunique()

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def get_wrong_predictions(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.asarray([(p, l) for (p, l) in zip(pred_flat, labels_flat) if p != l])

def get_sents_indices(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.asarray([pred_ind for (pred_ind, pred_l) in enumerate(pred_flat) if pred_l != labels_flat[pred_ind]])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(3)

# Load test data
test_labels = torch.load("./test_data/tradr/"+test_name+"_label_tensors_class_81_len_45.pt")
test_data = torch.load("./test_data/tradr/"+test_name+"_text_tensors_class_81_len_45.pt")
test_att = torch.load("./test_data/tradr/"+test_name+"_attention_tensors_class_81_len_45.pt") 
test_pos = torch.load("./test_data/tradr/"+test_name+"_position_tensors_class_81_len_45.pt")
test_beta = torch.load("./test_data/tradr/"+test_name+"_beta_tensors_class_81_len_45.pt")
test_pos_tags = torch.load("./test_data/tradr/"+test_name+"_pos_tags_tensors_class_81_len_45.pt")
test_sub = torch.load("./test_data/tradr/"+test_name+"_subword_tensors_class_81_len_45.pt")

# Select a batch size for loading tensors.
batch_size = 16

# Create an iterator of our data with torch DataLoader.
test = TensorDataset(test_data, test_att, test_pos, test_beta, test_pos_tags, test_sub, test_labels)
test_sampler = RandomSampler(test)
test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=batch_size)

# Load the model
class BertFrameClassifier(torch.nn.Module):
  def __init__(self, num_labels, dropout=0.1):
    super(BertFrameClassifier, self).__init__()
    self.model = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=False, output_attentions=False)
    self.dropout = torch.nn.Dropout(p=dropout)
    self.linear_1 = torch.nn.Linear(1548, 773)
    self.tanh = torch.nn.Tanh()
    self.linear_2 = torch.nn.Linear(773, num_labels)
    self.embed_pos = torch.nn.Embedding(21, 4)
    self.embed_subword = torch.nn.Embedding(2, 1)
    # initialize weights in two additional layers
    torch.nn.init.xavier_uniform_(self.linear_1.weight)
    torch.nn.init.constant_(self.linear_1.bias, 0)
    torch.nn.init.xavier_uniform_(self.linear_2.weight)
    torch.nn.init.constant_(self.linear_2.bias, 0)
    torch.nn.init.xavier_uniform_(self.embed_pos.weight)
    torch.nn.init.xavier_uniform_(self.embed_subword.weight)

  def forward(self, text_tensors_batch, attention_tensors_batch, position_tensors_batch, beta_tensors_batch, pos_tags_tensors_batch, subword_tensors_batch):
    # Convert POS tags to embeddings
    looked_up_pos = self.embed_pos(pos_tags_tensors_batch) # B x L x 4
    # Convert subword tags to embeddings
    looked_up_subword = self.embed_subword(subword_tensors_batch) # B x L x 1
    # Get BERT model output
    last_hidden_state, _ = self.model(input_ids=text_tensors_batch, attention_mask=attention_tensors_batch)
    # Concatenate word embeddings and feature embeddings
    last_hidden_state = torch.cat((last_hidden_state, looked_up_pos, looked_up_subword), 2) # B x L x 773
    # Dropout
    last_hidden_state = self.dropout(last_hidden_state) # B x L x 768
    # Calculate target vectors for all sequences in the batch
    target_tensors_batch = torch.bmm(last_hidden_state.permute(0, 2, 1), position_tensors_batch.unsqueeze(2).float()) # B x 768 x 1    
    # Calculate temporary vectors that will serve as a base for alignment vectors
    temp_tensors_batch = torch.bmm(last_hidden_state, target_tensors_batch) # B x L x 1 
    # Replace all the values outside betas' range with '-inf' s.t. softmax outputs 0's for them
    betaed_temp_tensors_batch = temp_tensors_batch.masked_fill(beta_tensors_batch.unsqueeze(2).float() == 0, -1e18) # B x L x 1
    # Calculate alignment vectors for each sequence in the batch
    alignment_tensors_batch = torch.nn.functional.softmax(betaed_temp_tensors_batch, dim=1)
    # Calculate context vectors
    context_tensors_batch = torch.bmm(last_hidden_state.permute(0, 2, 1), alignment_tensors_batch)
    # Concatenate context and target vectors along the first dimension
    input_tensors_batch = torch.cat((context_tensors_batch, target_tensors_batch), 1).squeeze(2)
    linear_output_1 = self.linear_1(input_tensors_batch)
    tanh = self.tanh(linear_output_1)
    linear_output_2 = self.linear_2(tanh)
    return linear_output_2

frame_clf = BertFrameClassifier(num_labels=num_frames)

checkpoint = torch.load("./models/w10_hl12_pos_sub.pt", map_location="cuda:3")
frame_clf.load_state_dict(checkpoint['state_dict'])
frame_clf.to('cuda:3')

# Prediction on test set
print('Predicting labels for {:,} test sentences...'.format(len(test_data)))

# Put model in evaluation mode to evaluate loss on the test set
frame_clf.eval()

# Tracking variables 
predicted = []
true = []

pred_wrong = []
sents_wrong = []
att_masks_wrong = []
pos_wrong = []

pbar = ProgressBar()

# Evaluate data for one epoch
for batch in pbar(test_dataloader):
  # Add batch to GPU
  batch = tuple(t.to('cuda:3') for t in batch)
  # Unpack the inputs from our dataloader
  input_ids_b, input_att_b, input_pos_b, input_beta_b, input_pos_tags_b, input_sub_b, labels_b = batch
  input_ids_b = input_ids_b.to('cuda:3')
  input_att_b = input_att_b.to('cuda:3')
  input_pos_b = input_pos_b.to('cuda:3')
  input_beta_b = input_beta_b.to('cuda:3')
  input_pos_tags_b = input_pos_tags_b.to('cuda:3')
  input_sub_b = input_sub_b.to('cuda:3')
  # Telling the model not to compute or store gradients, saving memory and speeding up validation
  with torch.no_grad():
    # Forward pass, calculate logit predictions
    logits = frame_clf(input_ids_b, input_att_b, input_pos_b, input_beta_b, input_pos_tags_b, input_sub_b)
    
  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  preds = np.argmax(logits, axis=1).flatten()
  labels_b = labels_b.to('cpu').numpy().flatten()

  predicted.extend(preds)
  true.extend(labels_b)

  wrong = get_wrong_predictions(logits, labels_b)
  wrong_ind = get_sents_indices(logits, labels_b)
  #print(wrong_ind, tmp_test_accuracy)

  pred_wrong.append(wrong)
  sents_wrong.extend([input_ids_b[i].cpu().numpy() for i in wrong_ind]) # get id sequences that were classified wrong
  att_masks_wrong.extend([input_att_b[i].cpu().numpy() for i in wrong_ind]) # get the corresponding attention masks
  pos_wrong.extend([input_pos_b[i].cpu().numpy() for i in wrong_ind]) # get the corresponding targets


true = [ref_dict[i] for i in true]
predicted = [ref_dict[i] for i in predicted]

print("Making classification report...")
#print(classification_report_imbalanced(true, predicted, labels=target_names))
print(classification_report_imbalanced(true, predicted))

def truncate_padding(tpl):
    padded = tpl[0]
    mask = tpl[1]
    return [ind for (ind, m) in zip(padded, mask) if m == 1] # a list of bert ids for a given sequence

bert_ids_wrong = [truncate_padding(tpl) for tpl in zip(sents_wrong, att_masks_wrong)]
targets_wrong = [truncate_padding(tpl) for tpl in zip(sents_wrong, pos_wrong)]

# Convert ids to tokens
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
sents = [tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True) for ids in bert_ids_wrong]
targets = [tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True) for ids in targets_wrong]

pred_wrong = [arr for arr in pred_wrong if arr.size != 0] # get rid of empty arrays in pred_wrong
pred_wrong_flat = np.concatenate(pred_wrong)
with open("./w10_hl12_pos_sub_"+test_name+"_wrong_predictions.txt", "w") as f:
  f.write("predicted"+"\t"+"true"+"\t"+"target"+"\t"+"sent"+"\n")
  for (tpl, t, s) in zip(pred_wrong_flat, targets, sents):
    t = " ".join(t)
    s = " ".join(s)
    pred = ref_dict[tpl[0]]
    gold = ref_dict[tpl[1]]
    f.write("\t".join([pred, gold, t, s])+"\n")
