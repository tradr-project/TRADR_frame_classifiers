# A script that loads the TRADR training data and trains a PAFIBERT model
# as described in the paper "Positional Attention-based Frame Identification with BERT: 
# A Deep Learning Approach to Target Disambiguation and Semantic Frame Selection." by S. Tan and J. Na.
# The model uses information about targets or context, as well as additional lexical features, such as
# POS tags and subword masks of the input tokens. The model learns to differentiate between 81 frame labels. 
# The script relies on the tutorial by Chris McCormick: https://mccormickml.com/2019/07/22/BERT-fine-tuning/

import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, AutoModel
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import precision_recall_fscore_support
from imblearn.metrics import geometric_mean_score, make_index_balanced_accuracy
from progressbar import ProgressBar
import time
import datetime
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_iba(preds, labels):
  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  geo_mean = geometric_mean_score(labels_flat, pred_flat, average=None, sample_weight=None)
  iba_gmean = make_index_balanced_accuracy(alpha=0.1, squared=True)(geometric_mean_score)
  iba = iba_gmean(labels_flat, pred_flat, average=None, sample_weight=None)
  _, _, _, support = precision_recall_fscore_support(labels_flat, pred_flat, average=None, sample_weight=None)
  res = np.average(iba, weights=support)
  return res

def format_time(elapsed):
  # Takes a time in seconds and returns a string hh:mm:ss
  # Round to the nearest second.
  elapsed_rounded = int(round((elapsed)))
  # Format as hh:mm:ss
  return str(datetime.timedelta(seconds=elapsed_rounded))

best_iba = 0
ibas = []

class BertFrameClassifier(torch.nn.Module):
  def __init__(self, num_labels, dropout=0.1):
    super(BertFrameClassifier, self).__init__()
    self.model = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=False, output_attentions=False).to('cuda:1')
    self.input_embed = self.model.get_input_embeddings()
    self.dropout = torch.nn.Dropout(p=dropout)
    self.linear_1 = torch.nn.Linear(1546, 773).to('cuda:2')
    self.tanh = torch.nn.Tanh()
    self.linear_2 = torch.nn.Linear(773, num_labels).to('cuda:3')
    self.embed_pos = torch.nn.Embedding(21, 4).to('cuda:1')
    self.embed_subword = torch.nn.Embedding(2, 1).to('cuda:1')
    # initialize weights in additional layers
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
	# Get BERT input representations
    last_hidden_state, _ = self.model(inputs_embeds=looked_up_input, attention_mask=attention_tensors_batch)
    # Concatenate embeddings
    last_hidden_sate = torch.cat((last_hidden_state, looked_up_pos, looked_up_subword), 2)
    # Dropout
    last_hidden_state = self.dropout(last_hidden_state)
    # Calculate target vectors for all sequences in the batch
    target_tensors_batch = torch.bmm(last_hidden_state.permute(0, 2, 1), position_tensors_batch.unsqueeze(2).float()) # B x 773 x 1    
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
    linear_output_1 = self.linear_1(input_tensors_batch.to('cuda:2'))
    tanh = self.tanh(linear_output_1)
    linear_output_2 = self.linear_2(tanh.to('cuda:3'))
    return linear_output_2

# Load data
label_tensors = torch.load("./train_data/tradr/tradr_label_tensors_class_81_len_45.pt")
text_tensors= torch.load("./train_data/tradr/tradr_text_tensors_class_81_len_45.pt")
attention_tensors = torch.load("./train_data/tradr/tradr_attention_tensors_class_81_len_45.pt")
position_tensors = torch.load("./train_data/tradr/tradr_position_tensors_class_81_len_45.pt")
beta_tensors = torch.load("./train_data/tradr/tradr_beta_tensors_class_81_len_45.pt")
pos_tags_tensors = torch.load("./train_data/tradr/tradr_pos_tags_tensors_class_81_len_45.pt")
subword_tensors = torch.load("./train_data/tradr/tradr_subword_tensors_class_81_len_45.pt")

# Split data into 5 folds to perform 5-fold cross validation
skf = StratifiedKFold(n_splits=5)

for train_idx, test_idx in skf.split(text_tensors, label_tensors):
  train_data, test_data = text_tensors[train_idx], text_tensors[test_idx]
  train_att, test_att = attention_tensors[train_idx], attention_tensors[test_idx]
  train_labels, test_labels = label_tensors[train_idx], label_tensors[test_idx]
  train_pos, test_pos = position_tensors[train_idx], position_tensors[test_idx]
  train_beta, test_beta = beta_tensors[train_idx], beta_tensors[test_idx]
  train_pos_tags, test_pos_tags = pos_tags_tensors[train_idx], pos_tags_tensors[test_idx]
  train_sub, test_sub = subword_tensors[train_idx], subword_tensors[test_idx]

  # Select a batch size for loading tensors.
  batch_size = 8

  # Create an iterator of our data with torch DataLoader.
  train = TensorDataset(train_data, train_att, train_pos, train_beta, train_pos_tags, train_sub, train_labels)
  train_sampler = RandomSampler(train)
  train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=batch_size)
  
  test = TensorDataset(test_data, test_att, test_pos, test_beta, test_pos_tags, test_sub, test_labels)
  test_sampler = SequentialSampler(test)
  test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=batch_size)

  # Initialize the frame classifier
  frame_clf = BertFrameClassifier(num_labels=81)
  #frame_clf = frame_clf.cuda()

  # Number of training epochs
  epochs = 8

  total_steps = len(train_dataloader) * epochs
  learning_rate = 3e-5
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = AdamW(frame_clf.parameters(), lr=learning_rate, eps=1e-8)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

  # Set the seed value all over the place to make this reproducible.
  seed_val = 42

  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

  # Measure the total training time for the whole run.
  total_t0 = time.time()

  # Training
  # trange is a tqdm wrapper around the normal python range
  for epoch_i in range(epochs):
    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    t0 = time.time()

    # Set our model to training mode (as opposed to evaluation mode)
    frame_clf.train()

    # Tracking variables
    tr_loss = 0
    nb_tr_steps = 0
  
    pbar = ProgressBar()

    # Train the data for one epoch
    for batch in pbar(train_dataloader):

      # Add batch to GPU
      batch = tuple(t.to('cuda:1') for t in batch)
      # Unpack the inputs from our dataloader
      input_ids_b, input_att_b, input_pos_b, input_beta_b, input_pos_tags_b, input_sub_b, labels_b = batch
      input_ids_b = input_ids_b.to('cuda:1')
      input_att_b = input_att_b.to('cuda:1')
      input_pos_b = input_pos_b.to('cuda:1')
      input_beta_b = input_beta_b.to('cuda:1')
      input_pos_tags_b = input_pos_tags_b.to('cuda:1')
      input_sub_b = input_sub_b.to('cuda:1')
      #input_speaker_b = input_speaker_b.to('cuda:1')
      labels_b = labels_b.to('cuda:3')
      # Clear out the gradients (by default they accumulate)
      optimizer.zero_grad()
      # Forward pass
      labels_b_pred = frame_clf(input_ids_b, input_att_b, input_pos_b, input_beta_b, input_pos_tags_b, input_sub_b) 
      loss = loss_fn(labels_b_pred, labels_b)
      # Backward pass
      loss.backward()
      # Clip the norm of the gradients to 1.0.
      # This is to help prevent the "exploding gradients" problem.
      torch.nn.utils.clip_grad_norm_(frame_clf.parameters(), 1.0)
      # Update parameters and take a step using the computed gradient
      optimizer.step()
      scheduler.step()

      # Update tracking variables
      tr_loss += loss.item()
      nb_tr_steps += 1

    print("Average train loss: {}".format(tr_loss/nb_tr_steps))
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("")
    print("  Training epcoh took: {:}".format(training_time))

  # Testing the newly trained model
  print("")
  print("Testing the model...")

  t0 = time.time()

  # Put model in evaluation mode to evaluate loss on the test set
  frame_clf.eval()

  # Tracking variables 
  eval_iba = 0
  nb_eval_steps = 0

  # Evaluate data for one epoch
  for batch in test_dataloader:
    # Add batch to GPU
    batch = tuple(t.to('cuda:1') for t in batch)
    # Unpack the inputs from our dataloader
    input_ids_b, input_att_b, input_pos_b, input_beta_b, input_pos_tags_b, input_sub_b, labels_b = batch
    input_ids_b = input_ids_b.to('cuda:1')
    input_att_b = input_att_b.to('cuda:1')
    input_pos_b = input_pos_b.to('cuda:1')
    input_beta_b = input_beta_b.to('cuda:1')
    input_pos_tags_b = input_pos_tags_b.to('cuda:1')
    input_sub_b = input_sub_b.to('cuda:1')
    #input_speaker_b = input_speaker_b.to('cuda:1')
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      logits = frame_clf(input_ids_b, input_att_b, input_pos_b, input_beta_b, input_pos_tags_b, input_sub_b)
    
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    labels_b = labels_b.to('cpu').numpy()

    #tmp_eval_accuracy = flat_accuracy(logits, labels_b)
    tmp_eval_iba = flat_iba(logits, labels_b)
    
    #eval_accuracy += tmp_eval_accuracy
    eval_iba += tmp_eval_iba
    nb_eval_steps += 1

  model_iba = eval_iba/nb_eval_steps
  print("Testing IBA: {}".format(model_iba))
  ibas.append(model_iba)

  # Save the trained frame classifier
  if model_iba > best_iba:
    checkpoint = {'state_dict': frame_clf.state_dict()}
    torch.save(checkpoint, "./models/w10_hl12_pos_sub.pt")

  # Measure how long the testing run took
  testing_time = format_time(time.time() - t0)
  print("  Testing took: {:}".format(testing_time))

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
print("Average IBA for 5 folds is", sum(ibas)/5)
