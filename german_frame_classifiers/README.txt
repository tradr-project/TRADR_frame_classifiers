##################################
#       Folder contains          #
##################################

* "data" forder for all raw data (not in the form of tensors)

* "train_data" folder for TRADR training data and SALSA data for sampling (as tensors)

* "test_data" folder for TRADR testing data (as tensors)

* "models" folder for models as checkpoints

* "clean_salsa.py" script to remove cases where identical sentences with identical targets are mapped to different frames

* "explore_SALSA.zip" a project to extract SALSA data

* "split_raw_data.py" script to split data into training/testing data and SALSA subset for sampling

* "input_preprocessing.py" script to convert TRADR training data to tensors

* "preprocessing_for_testing.py" script to convert TRADR testing sets and SALSA subset for sampling to tensors

* "frame_classifier_with_sampling.py" script to train the model

* "test_frame_classifier.py" script to test the model on testing sets

* "frame_clf.yml" conda YAML file to build conda environment

##################################
#       Request the data         #
##################################

To reproduce the experiments from scratch, please request the German data here:
TRADR: DFKI MLT Lab Talking Robots Group: ivana.kruijff@dfki.de or at https://www.dfki.de/en/web/research/research-departments/multilinguality-and-language-technology/tr-team/
       The file should be called
SALSA: http://www.coli.uni-saarland.de/projects/salsa/corpus/

NOTE: Please, create a folder to store the data, e.g., 'data', place the requested data here 
and modify the paths in the scripts accordingly, if necessary.

##################################
#   Recreate the environment     #
##################################

Recreate the environment using "frame_clf.yml" file with the following command: "conda env create -f frame_clf.yml"
Activate the environment with "conda activate frame_clf", download SpaCy model necessary for POS-tagging with the
following command: "python -m spacy download de_core_news_sm"

##################################
#    Read and convert the data   #
##################################

Extract SALSA data:
```````````````````
Note: We assume that the requested SALSA data is here: "SALSA_2.0/salsa-corpus-2.0/salsa_release.xml".
Import the project called "explore_SALSA.zip" to an IDE (e.g., Eclipse), you may need to edit the paths depending on where the input/output
will be saved. Run the file called "Salsa_extractor.java". The program outputs a file called "salsa_data_with_targets_inconsistent.txt".
It is inconsistent, as it contains targets that are mapped to different frames. To remove the inconsistencies and duplicates use 
the script "clean_salsa.py", which outputs the file called "salsa_data_with_targets_consistent.tsv".

Usage example: python	clean_salsa.py


Prepare data for training/testing the classifier:
`````````````````````````````````````````````````
1) Use "split_raw_data.py" script to split TRADR text data ("tradr_ger_from_conll_format.txt") into training and testing parts 
   (90% and 10% respectively), remove all rare frames (which have less than 5 instances), 
   filter out from SALSA data all frames that do not occur in TRADR training data.
   The script outputs 4 text files: "tradr_main.tsv", "tradr_test_a.tsv", "tradr_test_b.tsv" and "salsa_for_sampling.tsv"

   Usage example: python split_raw_data.py

2) Use "input_preprocessing.py" script to convert TRADR text training data into tensors

   Usage example: python input_preprocessing.py

3) Use "preprocessing_for_testing.py" script to convert TRADR text testing data to tensors.
   Provide a name tag to specify the text file to process (either "tradr_test_a" or "tradr_test_b")
   and destination folder, e.g., "./test_data"

   Usage example to convert TRADR testing sets: python preprocessing_for_testing.py ./data/tradr_test_a.tsv ./test_data

   The same script can be used to convert text SALSA data for sampling to tensors, which 
   should be saved as part of the training data. Provide a name tag "salsa_for_sampling" and destination folder "./train_data"

   Usage example to convert SALSA for sampling: python preprocessing_for_testing.py ./data/salsa_for_sampling.tsv ./train_data

#################################
#        Train a model          #
#################################

Train a model using "frame_classifier_with_sampling.py". The model uses a context window of 10 tokens to the left and 
to the right of the target, last hidden state of the BERT base model and 2375 examples sampled from SALSA for sampling subset.
It is trained using a 5-fold cross-validation procedure. The best model (judging by the IBA score) out of five is saved.
All folds are validated on TRADR data. Sampled SALSA examples are used only for training.

Usage example: python	frame_classifier_with_sampling.py

#################################
#       Test the model          #
#################################

The model can be tested on both TRADR testing sets using "test_frame_classifier.py" script.
Provide a name tag for testing set (either "tradr_test_a" or "tradr_test_b"). The script also outputs a file
containing incorrectly classified utterances.

Usage example: python test_frame_classifier.py	tradr_test_a

#################################
#          Results              #
#################################

Classification reports contain scores coming from the following metrics:

PRE: precision
REC: recall (true positive rate)
SPE: specificity (or true negative rate)
F1: F-score
GEO: geometric mean
IBA: index of balanced accuracy
SUP: support (number of testing examples)

The metrics come from Python "imbalanced-learn" library, rely on macro-average and can be used with imbalanced data.
