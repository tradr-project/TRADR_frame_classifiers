# A script that loads the TRADR training data and trains a BertForSequenceClasification model
# i.e. a simple fine-tuning of BERT BASE pre-trained model is performed.
# The model does not hove any information about targets or context and learns to differentiate between 81 frame labels. 
# The script relies on the tutorial by Chris McCormick: https://mccormickml.com/2019/07/22/BERT-fine-tuning/

import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.model_selection import StratifiedKFold
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

# Load data
label_tensors = torch.load("./train_data/tradr/tradr_label_tensors_class_81_len_45.pt")
text_tensors= torch.load("./train_data/tradr/tradr_text_tensors_class_81_len_45.pt")
attention_tensors = torch.load("./train_data/tradr/tradr_attention_tensors_class_81_len_45.pt") 

# Split data into 5 folds to perform 5-fold cross validation
skf = StratifiedKFold(n_splits=5)

for train_index, test_index in skf.split(text_tensors, label_tensors):
  train_data, test_data = text_tensors[train_index], text_tensors[test_index]
  train_att, test_att = attention_tensors[train_index], attention_tensors[test_index]
  train_labels, test_labels = label_tensors[train_index], label_tensors[test_index]

  # Select a batch size for loading tensors.
  batch_size = 16

  # Create an iterator of our data with torch DataLoader.
  train = TensorDataset(train_data, train_att, train_labels)
  train_sampler = RandomSampler(train)
  train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=batch_size)
  
  test = TensorDataset(test_data, test_att, test_labels)
  test_sampler = SequentialSampler(test)
  test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=batch_size)

  # Initialize the frame classifier
  frame_clf = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=81, output_attentions = False, output_hidden_states = False)
  frame_clf = frame_clf.cuda()

  # Number of training epochs
  epochs = 8

  total_steps = len(train_dataloader) * epochs
  learning_rate = 3e-5
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
      batch = tuple(t.cuda() for t in batch)
      # Unpack the inputs from our dataloader
      input_ids_b, input_att_b, labels_b = batch
      # Clear out the gradients (by default they accumulate)
      optimizer.zero_grad()
      # Forward pass
      loss, logits = frame_clf(input_ids_b, token_type_ids=None, attention_mask=input_att_b, labels=labels_b)  
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
    batch = tuple(t.cuda() for t in batch)
    # Unpack the inputs from our dataloader
    input_ids_b, input_att_b, labels_b = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      loss, logits = frame_clf(input_ids_b, token_type_ids=None, attention_mask=input_att_b, labels=labels_b)
    
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
    torch.save(checkpoint, "./models/baseline.pt")

  # Measure how long the testing run took
  testing_time = format_time(time.time() - t0)
  print("  Testing took: {:}".format(testing_time))

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
print("Average IBA for 5 folds is", sum(ibas)/5)
