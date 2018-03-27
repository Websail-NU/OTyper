If you use this code, please cite our paper:
'OTyper: A Neural Architecture for Open Named Entity Typing' in AAAI 2018


The data files of FIGER / MSH can be download at:
  http://downey-n1.cs.northwestern.edu/downloads/OTyper_data_aaai18/
  After download, put them in the same folder with the code (in OTyper).


How to duplicate the results in paper:
  To train the model,
  run command: python run.py FIGER / python run.py MSH

  To get the result of type AUC, after finished traning,
  run command: python get_CV_results.py FIGER / python get_CV_results.py MSH

  Features can be set on or off by setting corresponding parameters to 1 or 0 in
  function run_helper() / run_helper_MSH()


How to make your train your own data:
  You first need to implement a class, which will return data batch by batch.
  Please refer figer_data_multi_label_batcher.py / umls_data_batcher.py for
  templates.

  You then need to write the code to train your data using OTyper, you can copy
  most of the code in seen_type_dot_distance_label_matrix.py / umls_seen_type_dot_distance_label_matrix.py

  These two files use an integer to represent a type. To do N-fold-CV, you also need
  to create a txt file for your own data, like CV_output.txt / CV_output_MSH.txt.
  The format is:
  Training type ids for the 1st fold-CV
  Dev types ids for the 1st fold-CV
  Test types ids for the 1st fold-CV
  Training type ids for the 2nd fold-CV
  Dev types ids for the 2nd fold-CV
  Test types ids for the 2nd fold-CV
  Training type ids for the 3rd fold-CV
  Dev types ids for the 3rd fold-CV
  Test types ids for the 3rd fold-CV
  ...
  Until Nth

If you have more questions, please contact me via email: zys133@eecs.northwestern.edu
