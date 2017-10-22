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


def read_csv(file_path):
    entity_list = []

    with codecs.open(file_path, 'rU') as csvfile:
        reader = csv.reader((line.replace('\0','') for line in csvfile), delimiter=',')
        csv.field_size_limit(sys.maxsize)
        next(reader)
        for row in reader:
            v = entity(row)
            entity_list.append(v)

    return entity_list


def gen_type_feature():
    entity_file_path = './data/state_of_the_art_test_word_with_context.txt'
    output_file_path = './data/state_of_the_art_test_et_features'

    entities = get_entities(entity_file_path)
    pattern_dict = get_pattern_dict()
    label_dict = get_label_dict()

    output_array = np.zeros((len(entities), 113, 3))

    web_isa_class_dict = get_web_isa_class_dict()

    for i in range(0, len(entities)):
        if i % 10000 == 0:
            with open('temp.txt', 'w') as f:
                f.write('{}\n'.format(i))

        entity_parts = entities[i].split()

        counts = np.zeros(113)
        vec = np.zeros((113, 3))

        for part in entity_parts:
            if part in web_isa_class_dict:
                for e in web_isa_class_dict[part]:
                    if e.class_string in label_dict:
                        type_id = label_dict[e.class_string]
                        vec[type_id][0] += e.total_frequency
                        vec[type_id][1] += e.num_patterns
                        vec[type_id][2] += e.num_url
                        counts[type_id] += 1.0

        for j in range(0, 113):
            if counts[j] == 0.0:
                continue
            for k in range(0, 3):
                vec[j][k] = vec[j][k] / counts[j]

        output_array[i] = vec

    np.save(output_file_path, output_array)


def get_web_isa_class_dict():
    web_isa_classes = read_csv('./data/cached_webisa.txt')
    web_isa_class_dict = {}
    for e in web_isa_classes:
        if not e.entity_string in web_isa_class_dict:
            web_isa_class_dict[e.entity_string] = []
        web_isa_class_dict[e.entity_string].append(e)

    return web_isa_class_dict


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


def get_entities(entity_file_path):
    entities = []
    with open(entity_file_path, 'r') as f:
        for line in f:
            entities.append(line.replace('\n', '').lower())

    return entities


def get_pattern_dict():
    pattern_dict = {}
    count = 0
    with open('./data/all_patterns.txt', 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            pattern_dict[line] = count
            count += 1

    return pattern_dict


def temp():
    pass


if __name__ == "__main__":
    # read_all_csv('/websail/common/webisadb/ituples/')
    gen_type_feature()
    # read_all_csv('/websail/common/webisadb/ituples/')
