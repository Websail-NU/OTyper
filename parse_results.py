import os
import subprocess
import numpy as np
def tail(f, n, offset=0):
  # stdin,stdout = os.popen2("tail -n "+str(n)+" "+f)
  # stdin.close()
  # lines = stdout.readlines(); stdout.close()
  # ret = np.zeros((2,2))
  # ret[0][0] = float(lines[0].split()[-2])
  # ret[0][1] = float(lines[0].split()[-1])
  # ret[1][0] = float(lines[1].split()[-2])
  # ret[1][1] = float(lines[1].split()[-1])
  # return ret
  proc = subprocess.Popen(["tail", "-n" , str(n), f], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
  lines,err=proc.communicate()
  # print(lines)
  lines = lines.decode("utf-8").split('\n')
  ret = np.zeros((2,2))
  ret[0][0] = float(lines[0].split()[-2])
  ret[0][1] = float(lines[0].split()[-1])
  ret[1][0] = float(lines[1].split()[-2])
  ret[1][1] = float(lines[1].split()[-1])
  return ret

def parse():
    feature_flag = 1
    for model_name in ['ave','LSTM', 'attention']:
        print(model_name)
        for entity_type_feature_flag in range(0, 2):
            for exact_entity_type_feature_flag in range(0, 2):
                for type_only_feature_flag in range(0, 2):
                    for id_select_flag in range(2, 3):
                        log_path = './log_files/'
                        log_path += (model_name + '_')
                        log_path += (str(feature_flag) + '_')
                        log_path += (str(entity_type_feature_flag) + '_')
                        log_path += (str(exact_entity_type_feature_flag) + '_')
                        log_path += (str(type_only_feature_flag) + '_')
                        log_path += (str(id_select_flag))
                        log_path += '.txt'
                        result = tail(log_path, 2)
                        print('{}\t{}\t{}\t{}'.format(result[0][0],result[0][1],result[1][0],result[1][1]))

if __name__ == "__main__":
    parse()
