##################################
#       Folder contains          #
##################################

* "data" folder for all raw data (not in the form of tensors)

* "train_data" folder for TRADR training data and FrameNet data for sampling (as tensors):
	- "framenet" folder for FrameNet training data (as tensors)
	- "tradr" folder for TRADR training data and FrameNet data for sampling

* "test_data" folder for TRADR and FrameNet testing data (as tensors):
	- "framenet" folder for FrameNet testing data and TRADR testing set C preprocessed to be compatible with
	  the model trained on FrameNet training data (i.e. considering 931 classes and max. length of the sequence of 314 tokens).
	- "tradr" folder for TRADR testing data

* "models" folder for models as a checkpoints

* "additional_training_scripts" folder with scripts that allow to train other versions of the frame classifier, e.g.,
   versions with sampling and/or other features, such as speaker and dialogue act tags.
	- "frame_classifier_extra_discourse_features.py": shows how to integrate speaker and dialodue act tags as features
	  (embeddings of these features are concatenated with vectors representing targets and contexts)
	- "frame_classifier_extra_features_with_filtering.py": shows how to integrate POS tags and subword masks as features,
	  as well as frame filtering mechanism
	- "frame_classifier_with_sampling.py": shows how to perform three types of sampling (blind, balancing and equal)

* "split_raw_data.py" script to split TRADR and FrameNet data into training/testing data and create FrameNet subset for sampling

* "input_preprocessing.py" script to convert TRADR or FrameNet training data to tensors

* "preprocessing_for_testing_framenet.py" script to convert FrameNet testing data and TRADR testing set C to tensors

* "preprocessing_for_testing_tradr.py" script to convert TRADR testing sets and FrameNet data for sampling to tensors

* "baseline.py" script to train the baseline BertForSequenceClassification model

* "frame_classifier_fn.py" script to train a PAFIBERT frame classifier on the FrameNet data

* "basic_model.py" script to train a PAFIBERT frame classifier on TRADR data

* "frame_classifier_extra_features.py" script to train a PAFIBERT frame classifier on TRADR data with two extra lexical features.

* "test_baseline.py" script to test the BertForSequenceClassification model on TRADR testing sets

* "test_frame_classifier_fn.py" script to test the PAFIBERT model trained on FrameNet data (can be tested on FrameNet and TRADR testing sets)

* "test_basic_model.py" script to test the PAFIBERT model rained on TRADR data (without extra features, but sampling is possible)

* "test_frame_classifier_extra_features.py" script to test the PAFIBERT model with extra features (POS tags and subword masks) on TRADR data

* "additional_testing_scripts" folder with scripts to test a couple of other versions of the frame classifier (see "additional_training_scripts" folder)
	- "test_frame_classifier_extra_discourse_features.py"
	- "test_frame_classifier_extra_features_with_filtering.py"

* "frame_clf.yml" conda YAML file to build conda environment

##################################
#   Request the English data     #
##################################

To reproduce the experiments from scratch, please request the English data from:
TRADR: DFKI MLT Lab Talking Robots Group: ivana.kruijff@dfki.de or at https://www.dfki.de/en/web/research/research-departments/multilinguality-and-language-technology/tr-team/
       The file should be called 'tradr_data_for_clf.tsv'
FrameNet: https://framenet.icsi.berkeley.edu/fndrupal/framenet_request_data (it is possible to extract all necessary information with NLTK framenet library)

NOTE: Please, create a folder to store the raw data, e.g., 'data', place the requested data here 
and modify the paths in the scripts accordingly, if necessary.

NOTE: The folders 'train_data' and 'test_data' should contain all necessary data as tensors.

##################################
#   Recreate the environment     #
##################################

Recreate the environment using "frame_clf.yml" file with the following command: "conda env create -f frame_clf.yml"
Activate the environment with "conda activate frame_clf", download SpaCy model necessary for POS-tagging with the
following command: "python -m spacy download en_core_web_md"

##################################
#    Read and convert the data   #
##################################

Extract FrameNet data:
``````````````````````
Use the script called "extract_framenet_data.py". The actual corpus is not really necessary.

Usage example: python	extract_framenet_data.py

Prepare data for training/testing the classifier:
`````````````````````````````````````````````````
1) Use "split_raw_data.py" script to split TRADR and FrameNet data into training and testing parts 
   (90% and 10% respectively), remove all rare frames (which have less than 5 instances), and to create
   a subset of FrameNet data suitable for sampling, i.e. a subset that contains only he instances of frames
   present in TRADR data.
   
   The script outputs 7 text files: "tradr_main.tsv", "tradr_test_a.tsv", "tradr_test_b.tsv", "tradr_test_c.tsv",
   "framenet_main.tsv", "framenet_test.tsv" and "framenet_for_sampling.tsv".
   "tradr_test_b.tsv" is a subset of "tradr_test_a.tsc" with all instances of frames 'Communication_by_protocol' and
   'Communication_response_message' removed. "tradr_test_c.tsv" is also a subset of "tradr_test_.tsv" and contains 
   the instances of frames common to both "tradr_main.tsv" and "framenet_main.tsv".

   Usage example: python split_raw_data.py

2) Use "input_preprocessing.py" script to convert text training data into tensors. The script can be used with
   both TRADR and FrameNet data, use either 'tradr' or 'framenet' argument to specify what data to pre-process.
   The script is not devised for the pre-processing of the subset of FrameNet data for sampling 
   (use "preprocessing_for_testing_tradr.py" for this purpose).
   Note that for FrameNet data the execution of the script takes a lot of time.

   Usage example: python input_preprocessing.py tradr

3) Use "preprocessing_for_testing_framenet.py" script to convert FrameNet testing data and TRADR testing set C to tensors.
   Provide a name tag to specify the text file to process (either 'tradr' or 'framenet'). The data preprocessing is performed
   considering 931 classes and max. sequence length of 314 tokens.

   Usage example to convert TRADR testing set C: python preprocessing_for_testing_framenet.py tradr
   Usage example to convert FrameNet testing set: python preprocessing_for_testing_framenet.py framenet

4) Use "preprocessing_for_testing_tradr.py" script to convert all TRADR testing sets or FrameNet data for sampling to tensors.
   Provide a name tag to specify the text file to process (either 'tradr' or 'framenet'). Note that FrameNet data for sampling
   is actually needed for training and will be stored in the "./train_data/tradr/" folder. The data preprocessing is performed
   considering 81 classes and max. sequence length of 45 tokens.

   Usage example to convert a TRADR testing set: python preprocessing_for_testing_tradr.py tradr_test_a
   Usage example to convert FrameNet for sampling set: python preprocessing_for_testing_tradr.py framenet


#################################
#        Train a model          #
#################################

We provide several scripts for training a frame classifier:

1) Use "baseline.py" to train a BertForSequenceClassification model, which assumes simple fine-tuning of
   the pre-trained BERT BASE model. The model is trained on TRADR data and learns to differentiate between 81 frame labels.
   The model does not take into consideration frame-evoking targets and their contexts. The script assumes that the model
   is to be trained and saved on GPU, edit the script to train and/or save on CPU.

   Usage example: python baseline.py

2) Use "frame_classifier_fn.py" to train a PAFIBERT BASE model on FrameNet data. The model relies on the information about
   targets and their contexts. It does not consider extra features, such as POS tags and/or subword masks.
   The script assumes that the model is to be trained and saved on GPU, edit the script to train and/or save on CPU.

   Usage example: python frame_classifier_fn.py

3) Use "basic_model.py" to train a PAFIBERT BASE model on TRADR data. The model relies on the information about
   targets and their contexts. It does not consider sampling additional instances from FrameNet data or using extra features, 
   such as POS tags and/or subword masks.
   The script assumes that the model is to be trained and saved on GPU, edit the script to train and/or save on CPU.

   Usage example: python basic_model.py

4) Use "frame_classifier_extra_features.py" to train a PAFIBERT BASE model on TRADR data (no additional sampling).
   The model uses two extra features: POS tags and subword masks of the input tokens (lexical features). This model
   demonstrated the best IBA score when evaluated on TRADR testing sets A, B and C.

   Usage example: python frame_classifier_extra_features.py

All models utilizing targets and their contexts use a context window of 10 tokens to the left and 
to the right of the target and last hidden state of the BERT base model.
They are trained using a 5-fold cross-validation procedure. The best model (judging by IBA score) out of five is saved.
All folds are validated on TRADR data.

#################################
#       Test the model          #
#################################

1) Use "test_baseline.py" script to test baseline model "baseline.pt" (trained with "baseline.py" script). Please, provide
   the testing set flag as argument: either 'tradr_test_a' or 'tradr_test_b' or 'tradr_test_c'
   The classification report is saved in the "classification_reports" folder under the name [model_name]_[testing_data]_wrong_predictions.txt

   Usage example: python test_baseline.py tradr_test_a

2) Use "test_frame_classifier_fn.py" script to test "frame_classifier_fn.pt" model (trained with "frame_classifier_fn.py script).
   Provide either 'tradr_test_c' or 'framenet_test' tag as argument.
   The classification report is saved in the "classification_reports" folder under the name [model_name]_[testing_data]_wrong_predictions.txt

   Usage example: python test_frame_classifier_fn.py framenet_test

3) Use "test_basic_model.py" script to test "basic_model.pt" model (trained with "basic_model.py" script). Please, provide
   the testing set flag as argument: either 'tradr_test_a' or 'tradr_test_b' or 'tradr_test_c'
   The classification report is saved in the "classification_reports" folder under the name [model_name]_[testing_data]_wrong_predictions.txt

   Usage example: python test_basic_model.py tradr_test_a

   Note that the script can be used to test models trained with additional sampling instances (but without any extra features or filtering).
   To do so, replace "basic_model.pt" in the line "checkpoint = torch.load("./models/basic_model.pt", map_location="cuda:3")" with 
   a suitable model.

4) Use "test_frame_classifier_extra_features.py" script to test "w10_hl12_pos_sub.pt" model (trained with "frame_classifier_extra_features.py" script).
   Please, provide the testing set flag as argument: either 'tradr_test_a' or 'tradr_test_b' or 'tradr_test_c'.
   The classification report is saved in the "classification_reports" folder under the name [model_name]_[testing_data]_wrong_predictions.txt

   Usage example: python test_frame_classifier_extra-features.py tradr_test_a

##################################
#       Used data sizes          #
##################################

-----------------------------------------------------------------------------------
 Data                                    | TRADR | # frames | FrameNet | # frames | 
-----------------------------------------------------------------------------------
 Size before dropping dupes              | 3,521 |    190   |  200,750 |   1,014  |
-----------------------------------------------------------------------------------
 Size after dropping dupes               | 2,930 |    190   |  199,509 |   1,024  |
-----------------------------------------------------------------------------------
 Main (training + validation) data size  | 2,637 |    184   |  179,559 |   1,012  | 
-----------------------------------------------------------------------------------
 Main data size (rare frames removed)    | 2,444 |     81   |  179,386 |    931   |
-----------------------------------------------------------------------------------
 Testing data size (initial)             |  293  |     -    |   19,950 |     -    |
-----------------------------------------------------------------------------------
 Testing data size (rare frames removed) |  268  |     -    |   19,923 |    931   |
-----------------------------------------------------------------------------------
 Testing set A                           |  268  |     81   |     -    |     -    |
-----------------------------------------------------------------------------------
 Testing set B                           |  247  |     79   |     -    |     -    | 
-----------------------------------------------------------------------------------
 Testing set C                           |  234  |     50   |     -    |     -    |
-----------------------------------------------------------------------------------

#################################
#          Results              #
#################################

Classification reports contain scores coming from the following metrics:

PRE: precision
REC: recall (true positive rate)
SPE: specificity (or true negative rate)
F1:  F-score
GEO: geometric mean
IBA: index of balanced accuracy
SUP: support (number of testing examples)

The metrics come from Python "imbalanced-learn" library, rely on macro-average and can be used with imbalanced data.
