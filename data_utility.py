import numpy as np
import gensim
import random
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
import figer_data_multi_label_batcher
from collections import defaultdict
# import cPickle as pickle
import pickle
import evaluate
import seen_type_dot_distance_label_matrix
import sys
from sklearn.metrics import roc_auc_score
# from get_CV_results import get_one_CV_results_test_F1s


class type_f1s:
    def __init__(self):

        self.train_f1_dict = {}
        self.dev_f1_dict = {}
        self.test_f1_dict = {}

        self.train_F1_num = []
        self.dev_F1_num = []
        self.test_F1_num = []

        self.test_auc_dict = {}

        self.test_trues = None
        self.test_scores = None



    def add_f1s(self, id_list, f1_list, F1_num, flag):
        for i in range(0, len(id_list)):
            if flag == 0:
                self.train_f1_dict[id_list[i]] = f1_list[i]
            elif flag == 1:
                self.dev_f1_dict[id_list[i]] = f1_list[i]
            elif flag == 2:
                self.test_f1_dict[id_list[i]] = f1_list[i]

        if flag == 0:
            self.train_F1_num = F1_num
        elif flag == 1:
            self.dev_F1_num = F1_num
        elif flag == 2:
            self.test_F1_num = F1_num


    def add_test_auc(self, id_list, trues, scores):
        self.test_trues = trues
        self.test_scores = scores

        trues = np.transpose(trues)
        scores = np.transpose(scores)

        #find the reason why this happen
        for i in range(0, len(id_list)):
            if sum(trues[i]) == 0:
                self.test_auc_dict[id_list[i]] = 0.0
            else:
                self.test_auc_dict[id_list[i]] = roc_auc_score(trues[i], scores[i])




#   input: the txt figer data
#   output: Words.txt  Type.npy

def read_Mentions_embedding_types():

    with open('all_types.txt', 'r') as f:
        all_types = [data.replace('\n','') for data in f]
    with open('held_out_types.txt', 'r') as f:
        held_out_types = [data.replace('\n','') for data in f]
    # w2v = gensim.models.KeyedVectors.load_word2vec_format('/websail/common/embeddings/glove/840B/glove.840B.300d.bin', binary=True)

    Words = []
    Types = []
    Contexts = []
    Features = []
    dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')

    with open('/websail/common/figer_data/state_of_the_art_dev_data.txt', 'r') as f:
        count = 0
        for line in f:
            if count % 6 ==0:
                raw_entity_string = line.replace('\n', '')
            if count % 6 == 1:
                context_string = line.replace('\n', '')
            if count % 6 == 2:
                start = int(line)
            if count % 6 == 3:
                end = int(line)
                entity_refined_string = get_entity_string(raw_entity_string, start, end)
            if count % 6 == 4:
                feature_array = line.split()
                feature = []
                for e in feature_array:
                    feature.append(int(e))
            if count % 6 == 5:
                raw_labels_string = line.replace('\n', '')

                if not entity_refined_string is None:
                    tagged_context = get_tagged_context(context_string, start, end)

                    local_Types = np.zeros(113)
                    for e in raw_labels_string.split():
                        local_Types[dicts['label2id'][e]] = 1

                    Words.append(entity_refined_string)
                    Contexts.append(tagged_context)
                    Types.append(local_Types)
                    Features.append(feature)

            count += 1

    print(len(Words))
    print(len(Types))
    print(len(Contexts))
    print(len(Features))

    with open('data/state_of_the_art_dev_word_with_context.txt', 'w') as f:
        for word in Words:
            f.write("{}\n".format(word))
    with open('data/state_of_the_art_dev_tagged_context.txt', 'w') as f:
        for line in Contexts:
            f.write("{}\n".format(line))

    np.save('data/state_of_the_art_dev_Feature', Features)
    np.save('data/state_of_the_art_dev_Types_with_context', Types)


def get_all_unseen_types():
    all_test_id = []
    train_ids, dev_ids, test_ids = seen_type_dot_distance_label_matrix.get_CV_info('CV_output.txt')
    for i in range(0, 10):
        all_test_id += test_ids[i]

    return all_test_id


def gen_word_list():
    w2v = gensim.models.KeyedVectors.load_word2vec_format('/websail/common/embeddings/glove/840B/glove.840B.300d.bin', binary=True)
    word_set = set()
    word_set.add('_my_null_')
    word_set.add('unk')
    with open('/websail/common/figer_data/state_of_the_art_train_data.txt', 'r') as f:
        count = 0
        for line in f:
            if count % 5 ==0:
                raw_entity_string = line
                for e in raw_entity_string.split():
                    if e in w2v:
                        word_set.add(e)
            if count % 5 == 1:
                context_string = line
                for e in context_string.split():
                    if e in w2v:
                        word_set.add(e)
            if count % 5 == 2:
                pass
            if count % 5 == 3:
                pass
            if count % 5 ==4:
                raw_labels_string = line
                raw_labels_string = raw_labels_string.replace('/', ' ')
                raw_labels_string = raw_labels_string.replace('_', ' ')
                for e in raw_labels_string:
                    if e in w2v:
                        word_set.add(e)
            count += 1

    with open('/websail/common/figer_data/state_of_the_art_test_data.txt', 'r') as f:
        count = 0
        for line in f:
            if count % 5 ==0:
                raw_entity_string = line
                for e in raw_entity_string.split():
                    if e in w2v:
                        word_set.add(e)
            if count % 5 == 1:
                context_string = line
                for e in context_string.split():
                    if e in w2v:
                        word_set.add(e)
            if count % 5 == 2:
                pass
            if count % 5 == 3:
                pass
            if count % 5 ==4:
                raw_labels_string = line
                raw_labels_string = raw_labels_string.replace('/', ' ')
                raw_labels_string = raw_labels_string.replace('_', ' ')
                for e in raw_labels_string:
                    if e in w2v:
                        word_set.add(e)
            count += 1

    with open('data/word_list.txt', 'w') as f:
        for e in word_set:
            f.write("{}\n".format(e))

    with open('/websail/common/figer_data/state_of_the_art_dev_data.txt', 'r') as f:
        count = 0
        for line in f:
            if count % 5 ==0:
                raw_entity_string = line
                for e in raw_entity_string.split():
                    if e in w2v:
                        word_set.add(e)
            if count % 5 == 1:
                context_string = line
                for e in context_string.split():
                    if e in w2v:
                        word_set.add(e)
            if count % 5 == 2:
                pass
            if count % 5 == 3:
                pass
            if count % 5 ==4:
                raw_labels_string = line
                raw_labels_string = raw_labels_string.replace('/', ' ')
                raw_labels_string = raw_labels_string.replace('_', ' ')
                for e in raw_labels_string:
                    if e in w2v:
                        word_set.add(e)
            count += 1

    with open('data/word_list.txt', 'w') as f:
        for e in word_set:
            f.write("{}\n".format(e))


def get_entity_string(raw_entity_string, start, end):
    # tokens = raw_entity_string.replace(',', '')
    # tokens = raw_entity_string.replace('(', '')
    # tokens = raw_entity_string.replace(')', '')
    tokens = raw_entity_string.split(' ')

    a = []
    for i in range(0, min(len(tokens),(end-start))):
        a.append(tokens[i])

    return ' '.join(a)


def get_tagged_context(context_string, start, end):
    # context_string = context_string.replace(',', '')
    # context_string = context_string.replace('.', '')
    # context_string = context_string.replace('(', '')
    # context_string = context_string.replace(')', '')
    # context_string = context_string.split(' ')
    tokens = context_string.split(' ')

    a = []
    for i in range(0, len(tokens)):
        if i == start:
            a.append('<e>')
        a.append(tokens[i])
        if i == (end - 1):
            a.append('</e>')

    return ' '.join(a)

def get_labels(raw_labels_string, all_types):
    ret = set()

    for raw_e in raw_labels_string.split(' '):
        last_raw_label = raw_e.split('/')[-1]
        if last_raw_label in all_types:
            ret.add(last_raw_label)
    return ret

def gen_label_id_2_emb():
    dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')
    w2v = gensim.models.KeyedVectors.load_word2vec_format('/websail/common/embeddings/glove/840B/glove.840B.300d.bin', binary=True)

    label_id2emb = []

    for i in range(0,113):
        line = dicts['id2label'][i]

        label_emb = np.zeros(300)
        count = 0.0
        for word in line.split('/'):
            if len(word) == 0:
                continue
            if word == 'livingthing':
                word = 'living_thing'
            word_embs = get_emb_of_word(word, w2v)
            for e in word_embs:
                label_emb += e
                count += 1.0
        label_id2emb.append(label_emb / count)
    label_id2emb = np.asarray(label_id2emb, dtype=np.float32)
    with open('data/labelid2emb.pkl', 'w') as f:
        pickle.dump(label_id2emb, f)


def get_emb_of_word(word, w2v):
    ret = []
    if word in w2v:
        ret.append(w2v[word])
    else:
        if '_' in word:
            for e in word.split('_'):
                if e in w2v:
                    ret.append(w2v[e])
                else:
                    print('did not find')
                    print(e)
                    ret.append(w2v['unk'])
        else:
            print('did not find')
            print(word)
            ret.append(w2v['unk'])

    return ret


def get_top_k_labels():
    dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')
    with open('data/labelid2emb.pkl', 'r') as f:
        label_id2emb = pickle.load(f)
    for i in range(0, 113):
        print('id = {}, {}  nearest = '.format(i, dicts['id2label'][i]))
        get_top_k_ids(i, 3, label_id2emb)
        for e in get_top_k_ids(i, 3, label_id2emb):
            print(str(dicts['id2label'][e]))
        print('---')


def get_top_k_ids(index, top_k, label_id2emb, order = 'top'):
    l = []

    for i in range(0, 113):
        l.append([i, cosine_similarity(label_id2emb[index].reshape(1, -1), label_id2emb[i].reshape(1, -1))])

    l.sort(key=lambda row: row[1], reverse=True)
    ret = []
    for i in range(1, top_k + 1):
        if order == 'top':
            ret.append(l[i][0])
        elif order == 'bot':
            ret.append(l[-i][0])
        else:
            print('error arg!')
            sys.exit()


    return ret

def get_test_type_occurrence():
    figer_test = figer_data_multi_label_batcher.figer_data_multi_label(entity_file = 'data/state_of_the_art_test_word_with_context.txt', \
                    context_file = 'data/state_of_the_art_test_tagged_context.txt', \
                    feature_file = 'data/state_of_the_art_test_Feature.npy', \
                    entity_type_feature_file = 'data/state_of_the_art_test_et_features.npy',\
                    entity_type_exact_feature_file = 'data/state_of_the_art_test_exact_et_features.npy',\
                    type_file = 'data/state_of_the_art_test_Types_with_context.npy')

    batch_data = figer_test.next_batch(range(0, 113))

    a = np.sum(batch_data[-1], 0)

    b = np.zeros((113, 2))
    for i in range(0, 113):
        b[i][0] = i
        b[i][1] = a[i]

    dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')

    with open('data/test_type_count.pkl', 'w') as f:
        pickle.dump(b, f)

    b = b[b[:,1].argsort()[::-1]]

    for i in range(0, 113):
        if b[i][1]> 0.0:
            # print '{}\t{}\t{}'.format(dicts['id2label'][int(b[i][0])], int(b[i][0]), b[i][1])
            print('{}\t{}'.format(dicts['id2label'][int(b[i][0])], b[i][1]))


def get_hearst_feature_predict():
    figer_test = figer_data_multi_label_batcher.figer_data_multi_label(entity_file = 'data/state_of_the_art_test_word_with_context.txt', \
                    context_file = 'data/state_of_the_art_test_tagged_context.txt', \
                    feature_file = 'data/state_of_the_art_test_Feature.npy', \
                    entity_type_feature_file = 'data/state_of_the_art_test_et_features.npy',\
                    entity_type_exact_feature_file = 'data/state_of_the_art_test_exact_et_features.npy',\
                    type_file = 'data/state_of_the_art_test_Types_with_context.npy')

    # batch_data = figer_test.next_batch(range(0, 113))

    # figer_train = figer_data_multi_label_batcher.figer_data_multi_label()

    total_count = np.zeros((2,2))
    for i in range(0, 10):
        batch_data = figer_test.next_batch(range(0, 113))
        t_count = get_counts(batch_data)
        total_count += t_count

    print(total_count)


def get_counts(batch_data):
    ret_array = np.zeros((2,2))
    for i in range(0, batch_data[-1].shape[0]):
        for j in range(0, 113):
            if batch_data[-3][i][j][0] > 0:
                if batch_data[-1][i][j] > 0:
                    ret_array[1][1] += 1
                else:
                    ret_array[1][0] += 1
            else:
                if batch_data[-1][i][j] > 0:
                    ret_array[0][1] += 1
                else:
                    ret_array[0][0] += 1

    return ret_array


def get_train_type_freq():
    figer = figer_data_multi_label_batcher.figer_data_multi_label()

    total_count = np.zeros(113)
    for i in range(0, 200):
        Ys = figer.next_batch(range(0, 113))[-1]
        print(Ys.shape)

        for j in range(0, Ys.shape[0]):
            for k in range(0, 113):
                total_count[k] += Ys[j][k]

    with open('data/train_type_count.pkl', 'w') as f:
        pickle.dump(total_count, f)


def get_cos_sim_matrix():
    with open('data/labelid2emb.pkl', 'r') as f:
        label_id2emb = pickle.load(f)

    cos_sim_matrix = np.zeros((113, 113))
    for i in range(0, 113):
        for j in range(0, 113):
            cos_sim_matrix[i][j] = cosine_similarity(label_id2emb[i].reshape(1, -1), label_id2emb[j].reshape(1, -1))

    with open('data/cos_sim_matrix.pkl', 'w') as f:
        pickle.dump(cos_sim_matrix, f)


def get_ave_top_k_prior(test_ids, train_ids, k,):
    ret_sum = []
    for test_id in test_ids:
        with open('data/cos_sim_matrix.pkl', 'r') as f:
            cos_sim_matrix = pickle.load(f)
        distance = cos_sim_matrix[test_id]

        sort_ids = np.argsort(distance)[::-1]

        top_ids = []
        for i in sort_ids[1:]:
            if i in train_ids:
                top_ids.append(i)

        top_ids = top_ids[0:k]

        with open('data/train_type_count.pkl', 'r') as f:
            total_count = pickle.load(f)

        np.set_printoptions(suppress=True)
        total_count /= 200000

        t_sum = 0.0

        for e in top_ids:
            t_sum += total_count[e]

        t_sum /= k

        ret_sum.append(t_sum)

    return ret_sum


def type_freq_webisa_feature():

    all_test_id = []
    train_ids, dev_ids, test_ids = seen_type_dot_distance_label_matrix.get_CV_info('CV_output.txt')
    for i in range(0, 10):
        all_test_id += test_ids[i]

    print(all_test_id)

    with open('data/test_type_count.pkl', 'r') as f:
        b = pickle.load(f)

    figer = figer_data_multi_label_batcher.figer_data_multi_label()
    type_only = figer.next_batch(range(0, 113))[-2]

    dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')
    print('label, freq, pattern_matches, URL')
    for e in all_test_id:
        print('{}\t{}\t{:.4f}\t{:.4f}'.format(dicts['id2label'][e], b[e][1], type_only[0][e][0], type_only[0][e][2]))


def get_10_CV_similarity_loss():
    pass
    # with open('data/cos_sim_matrix.pkl', 'r') as f:
    #     cos_sim_matrix = pickle.load(f)
    #
    # with open('data/test_type_count.pkl', 'r') as f:
    #     b = pickle.load(f)
    # with open('data/train_type_count.pkl', 'r') as f:
    #     total_count = pickle.load(f)
    # with open('data/labelid2emb.pkl', 'r') as f:
    #     label_id2emb = pickle.load(f)
    #
    # supervised_auc_dict = get_CV_results.get_single_log_file_results(2)
    #
    # train_ids, dev_ids, test_ids = seen_type_dot_distance_label_matrix.get_CV_info('CV_output.txt')
    # for i in range(0, 10):
    #     seen_ids = train_ids[i] + dev_ids[i]
    #     unseen_ids = test_ids[i]
    #     for j in range(0, len(unseen_ids)):
    #         cos_sim = 0.0
    #         # for k in range(0, len(unseen_ids)):
    #         #     if k == j:
    #         #         continue
    #         #     cos_sim += cos_sim_matrix[unseen_ids[j]][unseen_ids[k]] * total_count[unseen_ids[k]]
    #         top_k_cos_dist = 0.0
    #         top_k_id = get_top_k_ids(unseen_ids[j], 7, label_id2emb, order = 'top')
    #
    #         top_k_auc = 0.0
    #         count = 0
    #         for k in range(0, 7):
    #             if top_k_id[k] in unseen_ids:
    #                 continue
    #             top_k_cos_dist += cos_sim_matrix[unseen_ids[j]][top_k_id[k]]
    #             top_k_auc += supervised_auc_dict[unseen_ids[j]]
    #             count += 1
    #             if count >= 3:
    #                 break
    #
    #         print '{}\t{}\t{:.4f}\t{:.4f}'.format(unseen_ids[j], b[unseen_ids[j]][1], top_k_cos_dist/3, top_k_auc/3)



def temp():
    global global_a
    global_a = 1
    print(global_a)






if __name__ == "__main__":
    # get_cos_sim_matrix()
    # get_hearst_feature_predict()
    # get_test_type_occurrence()
    # get_top_k_labels()
    temp()
    # read_Mentions_embedding_types()
    # get_train_type_freq()
    # get_ave_top_k_prior([0], range(0,113), 3)
    # type_freq_webisa_feature()
    # get_all_unseen_types()
    # get_10_CV_similarity_loss()
