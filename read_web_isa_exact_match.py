import csv
import json
import os
import sys
# import cPickle as pickle
import pickle
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
        self.pid_set = set()
        self.ipremods_list = []
        self.ipostmods_list = []
        self.cpremods_list = []
        self.cpostmods_list = []
        self.sub_frequency_list = []
        self.sub_url_list = []
        self.sub_pattern_list = []
        self.md_trans(json.loads(row[6]))


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

    with open(file_path, 'r') as raw_file:
        with codecs.open(file_path, 'rU') as csvfile:
            reader = csv.reader((line.replace('\0','') for line in csvfile), delimiter=',')
            csv.field_size_limit(sys.maxsize)
            next(reader)
            next(raw_file)
            for row in reader:
                v = entity(row, raw_row=next(raw_file))
                entity_list.append(v)

    return entity_list

def read_all_csv_umls(dir_path = '/websail/common/webisadb/ituples'):
    all_entity_str = set()
    with open('umls_data/refined_umls_word.txt', 'r') as f:
        for line in f:
            for e in line.strip().split():
                all_entity_str.add(e.lower())

    all_type_str = set()
    with open('umls_data/all_entity_name_list_new.pkl', 'rb') as f:
        umls_type_list = pickle.load(f)

    for t in umls_type_list:
        for e in t.strip().split():
            all_type_str.add(e.lower())

    all_entity_list = []
    count = 0
    with open('umls_data/cached_webisa.txt', 'w') as f:
        f.write('_id,instance,class,frequency,pidspread,pldspread,modifications\n')
    for file_name in os.listdir(dir_path):
        entity_list = read_csv_filter(os.path.join(dir_path, file_name), all_entity_str, all_type_str)
        # entity_list = read_csv('data/cached_webisa.txt', all_entity_str, all_type_str)
        with open('umls_data/cached_webisa.txt', 'a') as f:
            for e in entity_list:
                f.write(e.raw_row)
        # all_entity_list += entity_list
        count += 1
        if count % 10 == 0:
            print(count)

    return all_entity_list

def read_csv_filter(file_path, all_entity_str, all_type_str):
    entity_list = []

    with open(file_path, 'r') as raw_file:
        with codecs.open(file_path, 'rU') as csvfile:
            reader = csv.reader((line.replace('\0','') for line in csvfile), delimiter=',')
            csv.field_size_limit(sys.maxsize)
            next(reader)
            next(raw_file)
            for row in reader:
                v = entity(row, next(raw_file))
                if v.class_string in all_type_str and v.entity_string in all_entity_str:
                    entity_list.append(v)

    return entity_list


def gen_exact_matched_file():
    input_file_path = './data/cached_webisa.txt'
    output_file_path = './data/exact_matched_cached_webisa.txt'

    figer_entities_set = get_figer_entity_set()

    entity_list = read_csv(input_file_path)
    new_list = []
    for e in entity_list:
        if exact_matched(e, figer_entities_set):
            new_list.append(e)

    with open(output_file_path, 'w') as f:
        f.write('_id,instance,class,frequency,pidspread,pldspread,modifications\n')
        for e in new_list:
            f.write(e.raw_row)


def gen_exact_matched_file_umls():
    input_file_path = './umls_data/cached_webisa.txt'
    output_file_path = './umls_data/exact_matched_cached_webisa.txt'

    umls_entities_set = get_umls_entity_set()

    entity_list = read_csv(input_file_path)
    new_list = []
    for e in entity_list:
        if exact_matched(e, umls_entities_set):
            new_list.append(e)

    with open(output_file_path, 'w') as f:
        f.write('_id,instance,class,frequency,pidspread,pldspread,modifications\n')
        for e in new_list:
            f.write(e.raw_row)



def exact_matched(entity, figer_entities_set):
    for i in range(0, len(entity.cpremods_list)):
        array = []
        array.append(entity.entity_string.lower())
        if len(entity.ipremods_list[i]) != 0:
            array.append(entity.ipremods_list[i].lower())
        if len(entity.ipostmods_list[i]) != 0:
            array.append(entity.ipostmods_list[i].lower())

        if tuple(array) in figer_entities_set:
            return True

    return False


def get_umls_entity_set():
    umls_entities_set = set()
    with open('./umls_data/refined_umls_word.txt') as f:
        for line in f:
            line = line.replace('\n', '').lower()
            local_entity = []
            for e in line.split():
                if e.isalpha():
                    local_entity.append(e)

            local_entity.sort()
            umls_entities_set.add(tuple(local_entity))

    return umls_entities_set


def get_figer_entity_set():
    figer_entities_set = set()
    with open('./data/state_of_the_art_train_word_with_context.txt') as f:
        for line in f:
            line = line.replace('\n', '').lower()
            local_entity = []
            for e in line.split():
                if e.isalpha():
                    local_entity.append(e)

            local_entity.sort()
            figer_entities_set.add(tuple(local_entity))

    with open('./data/state_of_the_art_dev_word_with_context.txt') as f:
        for line in f:
            line = line.replace('\n', '').lower()
            local_entity = []
            for e in line.split():
                if e.isalpha():
                    local_entity.append(e)

            local_entity.sort()
            figer_entities_set.add(tuple(local_entity))

    with open('./data/state_of_the_art_test_word_with_context.txt') as f:
        for line in f:
            line = line.replace('\n', '').lower()
            local_entity = []
            for e in line.split():
                if e.isalpha():
                    local_entity.append(e)

            local_entity.sort()
            figer_entities_set.add(tuple(local_entity))

    return figer_entities_set


def gen_type_feature(entity_file_path, output_file_path):

    entities = get_entities(entity_file_path)
    # pattern_dict = get_pattern_dict()
    label_dict = get_label_dict()

    output_array = np.zeros((len(entities), 113, 3))

    web_isa_class_dict = get_web_isa_class_dict()

    for i in range(0, len(entities)):
        if i % 10000 == 0:
            with open('temp.txt', 'w') as f:
                f.write('{}\n'.format(i))

        counts = np.zeros(113)
        vec = np.zeros((113, 3))

        if entities[i] in web_isa_class_dict:
            for e in web_isa_class_dict[entities[i]]:
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


def gen_type_feature_umls(entity_file_path='umls_data/refined_umls_word.txt', output_file_path = 'umls_data/umls_exact_et_features'):
    entities = get_entities(entity_file_path)

    with open('umls_data/all_entity_name_list_new.pkl', 'rb') as f:
        umls_type_list = pickle.load(f)

    umls_type_list_lower = []
    for e in umls_type_list:
        umls_type_list_lower.append(e.lower())
    umls_type_list = umls_type_list_lower
    type_to_id_dict = build_type_lookup_table(umls_type_list)

    output_array = np.zeros((len(entities), 1387, 3))
    web_isa_class_dict = get_web_isa_class_dict(exact_matched_file_path='umls_data/exact_matched_cached_webisa.txt')

    for i in range(0, len(entities)):
        if i % 10000 == 0:
            with open('temp.txt', 'w') as f:
                f.write('{}\n'.format(i))

        counts = np.zeros(1387)
        vec = np.zeros((1387, 3))

        if entities[i] in web_isa_class_dict:
            for e in web_isa_class_dict[entities[i]]:
                c_m_string = e.class_string
                for i in range(0, len(e.cpremods_list)):
                    class_name = get_class_name(e.cpremods_list[i], c_m_string, e.cpostmods_list[i])
                    if class_name in type_to_id_dict:
                        type_id = type_to_id_dict[class_name]
                        vec[type_id][0] += e.total_frequency
                        vec[type_id][1] += e.num_patterns
                        vec[type_id][2] += e.num_url
                        counts[type_id] += 1.0

        for j in range(0, 1387):
            if counts[j] == 0.0:
                continue
            for k in range(0, 3):
                vec[j][k] = vec[j][k] / counts[j]

        output_array[i] = vec

    np.save(output_file_path, output_array)


def get_class_name(pre, mid, post):
    ret = ''

    if pre != '':
        ret = ' '.join([pre.lower(), mid.lower()])
    else:
        ret = mid

    if post != '':
        ret = ' '.join([ret, post.lower()])

    return ret


def build_type_lookup_table(umls_type_list):
    d = {}
    for i in range(0, len(umls_type_list)):
        d[umls_type_list[i].lower()] = i

    return d

def find_type_id_umls(class_string, umls_type_list):
    for i in range(0, len(umls_type_list)):
        if class_string in umls_type_list:
            return i

    return -1


def get_web_isa_class_dict(exact_matched_file_path='./data/exact_matched_cached_webisa.txt'):
    web_isa_classes = read_csv(exact_matched_file_path)
    web_isa_class_dict = {}
    for e in web_isa_classes:
        for i in range(0, len(e.cpostmods_list)):
            array = []
            array.append(e.entity_string.lower())
            if len(e.ipremods_list[i]) != 0:
                array.append(e.ipremods_list[i].lower())
            if len(e.ipostmods_list[i]) != 0:
                array.append(e.ipostmods_list[i].lower())

            array.sort()
            if not tuple(array) in web_isa_class_dict:
                web_isa_class_dict[tuple(array)] = []
            web_isa_class_dict[tuple(array)].append(e)

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
            line = line.replace('\n', '').lower()
            local_entity = []
            for e in line.split():
                if e.isalpha():
                    local_entity.append(e)

            local_entity.sort()
            entities.append(tuple(local_entity))

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
    a = set()
    with open('umls_data/all_entity_name_list_new.pkl', 'rb') as f:
        umls_type_list = pickle.load(f)

    for t in umls_type_list:
        for e in t.strip().split():
            a.add(e.lower())

    print(a)


if __name__ == "__main__":
    # # gen_exact_matched_file()
    # # gen_exact_matched_file_umls()
    # entity_file_path = './data/state_of_the_art_dev_word_with_context.txt'
    # output_file_path = './data/state_of_the_art_dev_exact_et_features'
    # gen_type_feature(entity_file_path, output_file_path)
    # entity_file_path = './data/state_of_the_art_test_word_with_context.txt'
    # output_file_path = './data/state_of_the_art_test_exact_et_features'
    # gen_type_feature(entity_file_path, output_file_path)
    # entity_file_path = './data/state_of_the_art_train_word_with_context.txt'
    # output_file_path = './data/state_of_the_art_train_exact_et_features'
    # gen_type_feature(entity_file_path, output_file_path)
    # # read_all_csv('/websail/common/webisadb/ituples/')
    # read_all_csv_umls()
    gen_type_feature_umls()
    # print(get_class_name('', 'haha', 'I-o'))
