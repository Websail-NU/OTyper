import pickle
import data_utility
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
from seen_type_dot_distance_label_matrix import get_CV_info
import parse_results
import numpy as np
import evaluate
from sklearn.metrics import roc_auc_score
import figer_data_multi_label_batcher
import argparse


def get_CV_results():
    folder = './type_f1_files/'
    model_string = 'attention_'
    flag_string = '1_0_1_1_'
    folder_log = './log_files/'
    with open('FIGER_data/test_type_count.pkl', 'rb') as f:
        b = pickle.load(f)

    final_test_dict = {}
    final_auc_dict = {}
    total_F1_num = np.zeros(3)
    example_f1s = []

    for i in range(10, 20):
        cv_id = i % 10

        cv = str(i)

        test_dict, test_F1_num, auc_dict = get_one_CV_results_test_F1s(folder, folder_log, model_string, flag_string, cv)

        final_test_dict.update(test_dict)
        final_auc_dict.update(auc_dict)
        total_F1_num += test_F1_num
        log_file_path = folder_log + model_string + flag_string + cv + '.txt'
        example_f1s.append(parse_results.tail(log_file_path, 2)[0][0])


    count_sum = 0
    for k in final_test_dict:
        count_sum += b[int(k)][1]

    type_F1_weighted = 0.0
    for k in final_test_dict:
        type_F1_weighted += (b[int(k)][1]/count_sum) * final_test_dict[k]


    print('FIGER(GOLD) OTyper type weighted average Macro {}'.format(type_F1_weighted))
    precision = total_F1_num[0] / total_F1_num[1]
    recall = total_F1_num[0] / total_F1_num[2]
    print('FIGER(GOLD) OTyper type micro {}'.format(evaluate.f1(precision, recall)))

    auc_array = []
    for  k in final_auc_dict:
        auc_array.append((b[int(k)][1]/count_sum) * final_auc_dict[k])
    print('FIGER(GOLD) OTyper Average type auc = {}'.format(np.sum(auc_array)))


    # print('---popularity---')
    # print('id---name---freq---auc---')
    # dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')
    # for k in final_auc_dict:
    #     print('{}\t{}\t{}\t{:.4f}'.format(k, dicts['id2label'][k], b[k][1], final_auc_dict[k]))


def get_CV_results_msh():

    folder = './umls_type_f1_files/'
    model_string = 'attention_'
    flag_string = '0_0_1_0_'
    folder_log = './umls_log_files/'
    with open('UMLS_data/test_type_count.pkl', 'rb') as f:
        b = pickle.load(f)

    final_test_dict = {}
    final_auc_dict = {}
    total_F1_num = np.zeros(3)
    example_f1s = []

    for i in range(10, 20):
        cv_id = i % 10

        cv = str(i)

        test_dict, test_F1_num, auc_dict = get_one_CV_results_test_F1s(folder, folder_log, model_string, flag_string, cv)

        final_test_dict.update(test_dict)
        final_auc_dict.update(auc_dict)
        total_F1_num += test_F1_num
        log_file_path = folder_log + model_string + flag_string + cv + '.txt'
        example_f1s.append(parse_results.tail(log_file_path, 2)[0][0])


    count_sum = 0
    for k in final_test_dict:
        count_sum += b[int(k)][1]

    type_F1_weighted = 0.0
    for k in final_test_dict:
        type_F1_weighted += (b[int(k)][1]/count_sum) * final_test_dict[k]


    print('MSH-WSD OTyper type weighted average Macro {}'.format(type_F1_weighted))
    precision = total_F1_num[0] / total_F1_num[1]
    recall = total_F1_num[0] / total_F1_num[2]
    print('MSH-WSD OTyper type micro {}'.format(evaluate.f1(precision, recall)))

    auc_array = []
    for  k in final_auc_dict:
        auc_array.append((b[int(k)][1]/count_sum) * final_auc_dict[k])
    print('MSH-WSD OTyper Average type auc = {}'.format(np.sum(auc_array)))





def get_type_sum_sim(test_type_id, seen_type_ids):
    with open('data/cos_sim_matrix.pkl', 'rb') as f:
        cos_sim_matrix = pickle.load(f)
    with open('data/train_type_count.pkl', 'rb') as f:
        total_count = pickle.load(f)

    sum_c = 0
    ret = 0.0
    for i in seen_type_ids:
        sum_c += total_count[i]
        ret += total_count[i]*cos_sim_matrix[i][test_type_id]

    ret /= sum_c

    return ret


def get_single_log_file_results(data_id):
    folder = './type_f1_files/'
    log_folder = './log_files/'
    model_string = 'attention_'
    flag_string = '1_0_1_1_'
    # dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')
    final_test_dict = {}
    data_id = str(data_id)
        # get_one_CV_results(folder, model_string, flag_string, cv, dicts)
    test_dict, F1_num, final_auc_dict = get_one_CV_results_test_F1s(folder, log_folder, model_string, flag_string, data_id)
    final_test_dict.update(test_dict)

    with open('data/test_type_count.pkl', 'rb') as f:
        b = pickle.load(f)

    count_sum = 0
    for k in final_test_dict:
        count_sum += b[int(k)][1]
    #
    # F1_weighted = 0.0
    # for k in final_test_dict:
    #     F1_weighted += (b[int(k)][1]/count_sum) * final_test_dict[k]
    #
    # print 'OpenNER single file type average Macro {}'.format(F1_weighted)


    auc_array = []
    for  k in final_auc_dict:
        auc_array.append((b[int(k)][1]/count_sum) * final_auc_dict[k])

    print('type auc = {:.4f}'.format(np.sum(auc_array)))

    return final_auc_dict

    # print '---popularity---'
    # print 'id---name---freq---auc---'
    # dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')
    # for k in final_auc_dict:
    #     print '{}\t{}\t{}\t{:.4f}'.format(k, dicts['id2label'][k], b[k][1], final_auc_dict[k])


def get_single_log_file_results_umls(data_id):
    # folder = './base_line_type_f1_files/'
    # log_folder = './base_line_log_files/'
    folder = './umls_type_f1_files/'
    log_folder = './umls_log_files/'
    model_string = 'attention_'
    # model_string = 'emb_sub_'
    flag_string = '0_0_1_0_'
    # dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')
    final_test_dict = {}
    data_id = str(data_id)
        # get_one_CV_results(folder, model_string, flag_string, cv, dicts)
    test_dict, F1_num, final_auc_dict = get_one_CV_results_test_F1s(folder, log_folder, model_string, flag_string, data_id)
    final_test_dict.update(test_dict)

    with open('umls_data/test_type_count.pkl', 'rb') as f:
        b = pickle.load(f)

    count_sum = 0
    for k in final_test_dict:
        count_sum += b[int(k)]
    #
    # F1_weighted = 0.0
    # for k in final_test_dict:
    #     F1_weighted += (b[int(k)][1]/count_sum) * final_test_dict[k]
    #
    # print 'OpenNER single file type average Macro {}'.format(F1_weighted)


    auc_array = []
    for  k in final_auc_dict:
        key = k
        auc_array.append((b[int(k)]/count_sum) * final_auc_dict[k])

    print('average auc = {} {:.4f}'.format(key, np.sum(auc_array)))

    return final_auc_dict

    # print '---popularity---'
    # print 'id---name---freq---auc---'
    # dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')
    # for k in final_auc_dict:
    #     print '{}\t{}\t{}\t{:.4f}'.format(k, dicts['id2label'][k], b[k][1], final_auc_dict[k])




def get_one_CV_results_test_F1s(folder_pkl, folder_log, model_string, flag_string, cv):
    file_path = folder_pkl + model_string + flag_string + cv + '.pickle'
    with open(file_path, 'rb') as data_file:
        data = pickle.load(data_file)
    file_path = folder_log + model_string + flag_string + cv + '.txt'
    best_id = get_best_epoch(file_path)
    return data[best_id].test_f1_dict, data[best_id].test_F1_num, data[best_id].test_auc_dict
    # return data[best_id].test_f1_dict, np.zeros(3)


def get_best_epoch(file_path):
    return -1
    count = 0
    best_f1 = -0.1
    ret_id = -1
    with open(file_path, 'r') as f:
        f.readline()
        for line in f:
            line = line.replace('\n','')
            if count % 9 == 5:
                t_f1 = line.split(' ')[-1]
                if float(t_f1) > best_f1:
                    best_f1 = float(t_f1)
                    ret_id = int(count / 9)
            count += 1

    return ret_id


def get_best_test_macro_F1(file_path):
    count = 0
    best_f1 = -0.1
    ret_id = -1
    with open(file_path, 'r') as f:
        f.readline()
        for line in f:
            line = line.replace('\n','')
            if count % 9 == 5:
                t_f1 = line.split(' ')[-1]
                if t_f1 > best_f1:
                    best_f1 = t_f1
                    ret_id = int(count / 9)
            count += 1

    line_num = ret_id * 9 + 7

    count = 0
    with open(file_path, 'r') as f:
        f.readline()
        for line in f:
            line = line.replace('\n','')
            if count == line_num:
                return line.split(' ')[-1]
            count += 1

    return -1


# similarity of cosin and f1
def get_one_CV_results(folder, model_string, flag_string, cv, dicts):
    with open('data/labelid2emb.pkl', 'rb') as f:
        label_id2emb = pickle.load(f)
    file_path = folder + model_string + flag_string + cv + '.pickle'
    with open(file_path, 'rb') as data_file:
        data = pickle.load(data_file)
        train_id_list = data[4].train_f1_dict.keys()
        for k, v in data[4].test_f1_dict.iteritems():
            similar_train_id = get_top_k_ids(k, 1, label_id2emb, train_id_list)
            print('cosine_sim= {:.4f} F1_sim= {:.4f}'.format(similar_train_id[0][1], \
                    F1_similarity(data[4].train_f1_dict[similar_train_id[0][0]], v)))


def F1_similarity(train_F1, test_F1):
    return 1.0 - abs(train_F1-test_F1)/(train_F1 + 1)


def get_top_k_ids(index, top_k, label_id2emb, train_id_list):
    l = []
    for i in train_id_list:
        l.append([i, cosine_similarity(label_id2emb[index].reshape(1, -1), label_id2emb[i].reshape(1, -1))[0][0]])

    l.sort(key=lambda row: row[1], reverse=True)

    ret = []

    for i in range(0, top_k):
        ret.append(l[i])

    return ret


def get_top_k_labels():
    dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')
    with open('data/labelid2emb.pkl', 'rb') as f:
        label_id2emb = pickle.load(f)
    for i in range(0, 113):
        print('id = {}, {}  nearest = '.format(i, dicts['id2label'][i]))
        get_top_k_ids(i, 3, label_id2emb)
        for e in get_top_k_ids(i, 3, label_id2emb):
            print(str(dicts['id2label'][e]))
        print('---')


def get_top_bot_performance():
    for i in range(0, 11):
        # get_single_log_file_results(i+30)
        get_single_log_file_results(i+90)

def loc_error_analysis():
    dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')
    with open('./type_f1_files/attention_0_0_0_0_4.pickle', 'rb') as data_file:
        data = pickle.load(data_file)

    print(data[4].test_scores.shape)
    print(data[4].test_trues.shape)
    print(roc_auc_score(np.transpose(data[4].test_trues)[0], np.transpose(data[4].test_scores)[0]))

    data[4].test_scores = np.reshape(data[4].test_scores, (563))

    figer_test = figer_data_multi_label_batcher.figer_data_multi_label(entity_file = 'data/state_of_the_art_test_word_with_context.txt', \
                    context_file = 'data/state_of_the_art_test_tagged_context.txt', \
                    feature_file = 'data/state_of_the_art_test_Feature.npy', \
                    entity_type_feature_file = 'data/state_of_the_art_test_et_features.npy',\
                    entity_type_exact_feature_file = 'data/state_of_the_art_test_exact_et_features.npy',\
                    type_file = 'data/state_of_the_art_test_Types_with_context.npy')

    batch_data = figer_test.next_batch(range(0,113))

    # print batch_data[-1][1]


    # score of location
    # entity name
    # context
    # corret labels


    vob = figer_data_multi_label_batcher.Vocabulary()
    indices = np.argsort(data[4].test_scores)[::-1]


    for i in range(0, len(indices)):
        print('---')
        for j in range(0, 113):
            if batch_data[-1][indices[i]][j] != 0.0:
                print(dicts['id2label'][j])
        for e in batch_data[0][indices[i]]:
            if vob.i2w(int(e)) == '_my_null_':
                continue
            print(vob.i2w(int(e)))
        if i >=102:
            break


    # for i in range(0, data[4].test_scores.shape[0]):
    #     print '---'
    #     for j in range(0, 113):
    #         if batch_data[-1][i][j] != 0.0:
    #             print j


def cap_ratio():
    np.set_printoptions(suppress=True)
    with open('data/train_type_count.pkl', 'rb') as f:
        total_count = pickle.load(f)

    print(total_count)
    vob = figer_data_multi_label_batcher.Vocabulary()
    figer = figer_data_multi_label_batcher.figer_data_multi_label()

    loc_cap_count = 0
    person_cap_count = 0
    for i in range(200):
        batch_data = figer.next_batch(range(0, 2))
        for j in range(0, batch_data[0].shape[0]):
            if batch_data[-1][j][0] != 0.0:
                if vob.i2w(int(batch_data[0][j][0]))[0].isupper():
                    loc_cap_count += 1
            if batch_data[-1][j][1] != 0.0:
                if vob.i2w(int(batch_data[0][j][1]))[0].isupper():
                    person_cap_count += 1
        # break
    print(loc_cap_count)
    print(person_cap_count)

def temp():
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_flag', help='which set of data to train', choices=['FIGER','MSH'])

    args = parser.parse_args()
    if args.data_flag == 'FIGER':
        get_CV_results()
    elif args.data_flag == 'MSH':
        get_CV_results_msh()
    else:
        print('unknown argument')
