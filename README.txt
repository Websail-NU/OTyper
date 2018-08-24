This repo is the code for OTyper.

If you use our code, please cite our paper:
'OTyper: A Neural Architecture for Open Named Entity Typing' in AAAI 2018

@inproceedings{Yuan2018OTyperAN,
  title={OTyper: A Neural Architecture for Open Named Entity Typing},
  author={Zheng Yuan and Doug Downey},
  booktitle={AAAI},
  year={2018}
}

The data files of FIGER / MSH can be download at:
  http://downey-n1.cs.northwestern.edu/downloads/OTyper_data_aaai18/
  Put FIGER_data and UMLS_data (MSH) in the same folder with the code (in OTyper).


How to duplicate the results in paper:
  To train the model,
  run command: python run.py FIGER 1 1 1 / python run.py MSH 0 1 0

  To get the result of type AUC, after finished traning,
  run command: python get_CV_results.py FIGER 1 1 1 / python get_CV_results.py MSH 0 1 0


Requirements:
  tensorflow 1.3.0
  python 3.6.0


How to train your own data:

  1. You first need to implement a class, which will return your data batch by batch.Please refer figer_data_multi_label_batcher.py / umls_data_batcher.py for templates.

  2. You then need to write the code to train your data using OTyper, you can copy most of the code in seen_type_dot_distance_label_matrix.py / umls_seen_type_dot_distance_label_matrix.py

  3. Write a function to get average type AUC, get_CV_results_figer() / get_CV_results_msh() are templates.

If you have more questions, please contact me via email: zys133@eecs.northwestern.edu
