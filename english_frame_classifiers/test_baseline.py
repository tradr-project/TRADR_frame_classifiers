# A script to evaluate the performance of the model
# on test data

import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertForSequenceClassification, BertTokenizerFast
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
num_frames = 81

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

# Select a batch size for loading tensors.
batch_size = 16

# Create an iterator of our data with torch DataLoader.
test = TensorDataset(test_data, test_att, test_labels)
test_sampler = RandomSampler(test)
test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=batch_size)

# Load the model
frame_clf = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = num_frames, output_attentions = False, output_hidden_states = False)
optimizer = AdamW(frame_clf.parameters())

checkpoint = torch.load("./models/baseline.pt", map_location="cuda:3")
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

pbar = ProgressBar()

# Evaluate data for one epoch
for batch in pbar(test_dataloader):
  # Add batch to GPU
  batch = tuple(t.to('cuda:3') for t in batch)
  # Unpack the inputs from our dataloader
  input_ids_b, input_att_b, labels_b = batch
  input_ids_b = input_ids_b.to('cuda:3')
  input_att_b = input_att_b.to('cuda:3')
  
  # Telling the model not to compute or store gradients, saving memory and speeding up validation
  with torch.no_grad():
    # Forward pass, calculate logit predictions
    outputs = frame_clf(input_ids_b, input_att_b)
  logits = outputs[0]
    
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

# Convert ids to tokens
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
sents = [tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True) for ids in bert_ids_wrong]

pred_wrong = [arr for arr in pred_wrong if arr.size != 0] # get rid of empty arrays in pred_wrong
pred_wrong_flat = np.concatenate(pred_wrong)
with open("./baseline_"+test_name+"_wrong_predictions.txt", "w") as f:
  f.write("predicted"+"\t"+"true"+"\t"+"sent"+"\n")
  for (tpl, s) in zip(pred_wrong_flat, sents):
    s = " ".join(s)
    pred = ref_dict[tpl[0]]
    gold = ref_dict[tpl[1]]
    f.write("\t".join([pred, gold, s])+"\n")
