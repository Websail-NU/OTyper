import cPickle as pickle
from sklearn.externals import joblib
import data_utility
import numpy as np
import seen_type_dot_distance_label_matrix

def gen_top_bot_file():
    # dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')
    with open('data/labelid2emb.pkl', 'r') as f:
        label_id2emb = pickle.load(f)

    top_14_types = [1, 2, 0, 3, 11, 8, 76, 20, 18, 5, 13, 24, 9, 38]

    # with open('top_3_14_types.txt', 'w') as f:
    #     for i in range(0, len(top_14_types)):
    #         test_type_id = top_14_types[i]
    #         top_k = data_utility.get_top_k_ids(test_type_id, 3, label_id2emb)
    #         train_ids, dev_ids, test_ids = gen_train_dev(top_14_types[i], top_k)
    #         f.write(' '.join(str(e) for e in train_ids))
    #         f.write('\n')
    #         f.write(' '.join(str(e) for e in dev_ids))
    #         f.write('\n')
    #         f.write(' '.join(str(e) for e in test_ids))
    #         f.write('\n')
    #
    # with open('bot_3_14_types.txt', 'w') as f:
    #     for i in range(0, len(top_14_types)):
    #         test_type_id = top_14_types[i]
    #         bot_k = data_utility.get_top_k_ids(test_type_id, 3, label_id2emb, order = 'bot')
    #         train_ids, dev_ids, test_ids = gen_train_dev(top_14_types[i], bot_k)
    #         f.write(' '.join(str(e) for e in train_ids))
    #         f.write('\n')
    #         f.write(' '.join(str(e) for e in dev_ids))
    #         f.write('\n')
    #         f.write(' '.join(str(e) for e in test_ids))
    #         f.write('\n')

    with open('random_3_14_types.txt', 'w') as f:
        for i in range(0, len(top_14_types)):
            test_type_id = top_14_types[i]
            top_k = data_utility.get_top_k_ids(test_type_id, 3, label_id2emb)
            bot_k = data_utility.get_top_k_ids(test_type_id, 3, label_id2emb, order = 'bot')
            must_in = top_k + bot_k
            print must_in
            candidates = []
            for j in range(0, 113):
                if (j in must_in) or (j == top_14_types[i]):
                    continue
                candidates.append(j)
            np.random.shuffle(candidates)
            train_ids, dev_ids, test_ids = gen_train_dev(top_14_types[i], candidates[0:3])
            f.write(' '.join(str(e) for e in train_ids))
            f.write('\n')
            f.write(' '.join(str(e) for e in dev_ids))
            f.write('\n')
            f.write(' '.join(str(e) for e in test_ids))
            f.write('\n')



def gen_train_dev(original_id, held_out_ids):
    rest_ids = []

    for i in range(0, 113):
        if (not i in held_out_ids) and (i != original_id):
            rest_ids.append(i)

    np.random.shuffle(rest_ids)
    dev_ids = rest_ids[0:10]
    train_ids = rest_ids[10:]

    return train_ids, dev_ids, [original_id]

def temp():
    with open('data/labelid2emb.pkl', 'r') as f:
        label_id2emb = pickle.load(f)
    top_14_types = [1, 2, 0, 3, 11, 8, 76, 20, 18, 5, 13, 24, 9, 38]
    train, dev, test = seen_type_dot_distance_label_matrix.get_CV_info('random_3_14_types.txt')
    for i in range(0, len(top_14_types)):
        test_type_id = top_14_types[i]
        top_k = data_utility.get_top_k_ids(test_type_id, 3, label_id2emb)
        bot_k = data_utility.get_top_k_ids(test_type_id, 3, label_id2emb, order = 'bot')
        must_in = top_k + bot_k

        l = train[i] + dev[i]
        for e in must_in:
            assert(e in l)



if __name__ == "__main__":
    # gen_top_bot_file()
    temp()
