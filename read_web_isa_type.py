import csv
import json
import os
import sys
import cPickle as pickle
import codecs
import linecache
from collections import defaultdict
from sklearn.externals import joblib
import numpy as np

class entity:
    def __init__(self, row, raw_row = None):
        self.raw_row = raw_row
        self.entity_string = row[1]
        self.class_string = row[2]
        self.total_frequency = int(row[3])
        self.num_patterns = int(row[4])
        self.num_url = int(row[5])
        # self.pid_set = set()
        # self.ipremods_list = []
        # self.ipostmods_list = []
        # self.cpremods_list = []
        # self.cpostmods_list = []
        # self.sub_frequency_list = []
        # self.sub_url_list = []
        # self.sub_pattern_list = []
        # self.md_trans(json.loads(row[6]))


    def md_trans(self, md):
        for e in md:
            self.cpremods_list.append(e['cpremod'])
            self.cpostmods_list.append(e['cpostmod'])
            self.ipremods_list.append(e['ipremod'])
            self.ipostmods_list.append(e['ipostmod'])
            self.sub_frequency_list.append(e['frequency'])

            local_pattern_list = []
            for s in e['pids'].split(';'):
                if s != '':
                    local_pattern_list.append(s)
                    self.pid_set.add(s)

            local_url_list = []
            for s in e['plds'].split(';'):
                if s != '':
                    local_url_list.append(s)

            self.sub_url_list.append(local_url_list)
            self.sub_pattern_list.append(local_pattern_list)


def read_csv(file_path, type_dict):
    ret_sum = np.zeros((113, 3))
    ret_count = np.zeros(113)

    with open(file_path, 'r') as raw_file:
        with codecs.open(file_path, 'rU') as csvfile:
            reader = csv.reader((line.replace('\0','') for line in csvfile), delimiter=',')
            csv.field_size_limit(sys.maxsize)
            next(reader)
            next(raw_file)
            for row in reader:
                v = entity(row, raw_row=next(raw_file))
                if v.class_string in type_dict:
                    t_id = type_dict[v.class_string]
                    ret_sum[t_id][0] += v.total_frequency
                    ret_sum[t_id][1] += v.num_patterns
                    ret_sum[t_id][2] += v.num_url
                    ret_count[t_id] += 1.0

    return ret_sum, ret_count


def gen_type_feature():
    output_file_path = './data/webisa_type_only_features'

    label_dict = get_label_dict()

    type_ave = read_all_csv('/websail/common/webisadb/ituples/', label_dict)

    np.save(output_file_path, type_ave)


def get_label_dict():
    dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')
    label_dict = {}

    count = 0
    for i in range(0, 113):
        str = dicts['id2label'][i].split('/')[-1]
        if '_' in str:
            str = dicts['id2label'][i].split('_')[-1]
        label_dict[str] = count
        count += 1

    return label_dict


def read_all_csv(dir_path, label_dict):
    type_sum = np.zeros((113, 3))
    type_count = np.zeros(113)
    for file_name in os.listdir(dir_path):
        t_sum, t_count = read_csv(os.path.join(dir_path, file_name), label_dict)
        type_sum += t_sum
        type_count += t_count

    type_ave = np.zeros((113, 3))

    for i in range(0, 113):
        if type_count[i] != 0.0:
            for j in range(0, 3):
                type_ave[i][j] = type_sum[i][j]/type_count[i]

    return type_ave


def temp():
    a = np.load( './data/webisa_type_only_features.npy')
    print a


if __name__ == "__main__":
    # temp()
    gen_type_feature()
