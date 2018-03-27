If you use our code to do research, please cite our paper 'OTyper: A Neural Architecture for Open Named Entity Typing' in AAAI 2018


How to duplicate the result in paper:
  To train the model,
  run command: python run.py FIGER / python run.py MSH

  To get the result of type AUC, after finished traning,
  run command: python get_CV_results.py FIGER / python get_CV_results.py MSH

  Features can be set on or off by setting corresponding parameters to 1 or 0 in function run_helper() / run_helper_MSH()
